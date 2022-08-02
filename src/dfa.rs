use flagset::{flags, FlagSet};
use petgraph::dot;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use super::{Assertions, Inst, Program};

#[derive(Clone, Debug)]
pub struct DFA {
    graph: Graph<Node, Edge>,
    initial: Box<[Rc<Vec<usize>>]>,
}

flags! {
    #[derive(Hash)]
    enum Constraint: u8 {
        /// any byte, or end of text
        Anything,
        /// [_0-9a-zA-Z]
        WordByte,
        /// [^_0-9a-zA-Z] or end of text
        NotWordByte,
        /// '\n' or end of text
        Newline,
        /// no byte is possible, only end of text
        EndText,
    }
}

impl Constraint {
    fn match_at_end(self) -> bool {
        !matches!(self, Constraint::WordByte)
    }

    fn ranges(self, lo: u8, hi: u8) -> impl Iterator<Item = (u8, u8)> {
        let ranges: &[(u8, u8)] = match self {
            Constraint::Anything => &[(0, 255)],
            Constraint::WordByte => &[
                // sorted in ASCII order to make the relationship with NotWordByte more clear
                (b'0', b'9'),
                (b'A', b'Z'),
                (b'_', b'_'),
                (b'a', b'z'),
            ],
            Constraint::NotWordByte => &[
                (0, b'0' - 1),
                (b'9' + 1, b'A' - 1),
                (b'Z' + 1, b'_' - 1),
                (b'_' + 1, b'a' - 1),
                (b'z' + 1, 255),
            ],
            Constraint::Newline => &[(b'\n', b'\n')],
            Constraint::EndText => &[],
        };

        ranges
            .iter()
            .copied()
            .map(move |(range_lo, range_hi)| (range_lo.max(lo), range_hi.min(hi)))
            .filter(|(range_lo, range_hi)| range_lo <= range_hi)
    }

    fn overlaps(self, lo: u8, hi: u8) -> bool {
        self.ranges(lo, hi).next().is_some()
    }

    fn intersection(self, other: Self) -> Option<Self> {
        use Constraint::*;
        match (self, other) {
            _ if self == other => Some(self),
            (Anything, _) => Some(other),
            (_, Anything) => Some(self),
            (WordByte, _) | (_, WordByte) => None,
            (EndText, _) | (_, EndText) => Some(EndText),
            (Newline, NotWordByte) | (NotWordByte, Newline) => Some(Newline),
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Debug)]
struct Thread {
    pc: u16,
    constraint: Constraint,
    saved: Rc<Vec<usize>>,
}

impl Thread {
    fn next(mut self) -> Self {
        self.pc += 1;
        self
    }
}

fn initial_states(program: &Program) -> Vec<Thread> {
    fn go(
        program: &Program,
        result: &mut Vec<Thread>,
        active: &mut Vec<FlagSet<Constraint>>,
        mut current: Thread,
    ) {
        let seen = &mut active[usize::from(current.pc)];
        if !seen.is_disjoint(current.constraint) {
            return;
        }
        *seen |= current.constraint;

        match program.buf[usize::from(current.pc)] {
            Inst::Range(..) | Inst::Match => result.push(current),
            Inst::Assertion(kind) => {
                let constraint = match kind {
                    // These assertions are trivially satisfied for the initial state.
                    Assertions::StartLine | Assertions::StartText => Constraint::Anything,
                    // These assertions apply to the following byte.
                    Assertions::EndLine => Constraint::Newline,
                    Assertions::EndText => Constraint::EndText,
                    // Start of text is treated like a NotWordByte.
                    Assertions::AsciiWordBoundary => Constraint::WordByte,
                    Assertions::AsciiNotWordBoundary => Constraint::NotWordByte,
                };
                if let Some(constraint) = current.constraint.intersection(constraint) {
                    current.constraint = constraint;
                    go(program, result, active, current.next());
                }
            }
            Inst::Save(reg) => {
                Rc::make_mut(&mut current.saved)[usize::from(reg)] = 0;
                go(program, result, active, current.next());
            }
            Inst::Jmp(to) => {
                current.pc = to;
                go(program, result, active, current);
            }
            Inst::Split { prefer_next, to } => {
                let mut next = current.clone().next();
                current.pc = to;
                if prefer_next {
                    std::mem::swap(&mut current, &mut next);
                }
                go(program, result, active, current);
                go(program, result, active, next);
            }
        }
    }

    let mut result = Vec::new();
    let mut active = vec![FlagSet::default(); program.buf.len()];
    let initial = Thread {
        pc: 0,
        constraint: Constraint::Anything,
        saved: Rc::new(vec![usize::MAX; program.registers.into()]),
    };
    go(program, &mut result, &mut active, initial);
    result
}

impl DFA {
    pub(crate) fn new(program: &Program) -> DFA {
        let initial_states = initial_states(program);
        let mut builder = Builder {
            program,
            states: HashMap::new(),
            worklist: Vec::new(),
            dfa: DFA {
                graph: Graph::new(),
                initial: initial_states
                    .iter()
                    .map(|thread| thread.saved.clone())
                    .collect(),
            },
            active: vec![FlagSet::default(); program.buf.len()],
            current_effects: Vec::new(),
            current_edges: BTreeMap::new(),
        };

        let initial_states: Box<[(u16, Constraint)]> = initial_states
            .into_iter()
            .map(|thread| (thread.pc, thread.constraint))
            .collect();

        builder.states.insert(
            initial_states.clone(),
            builder.dfa.graph.add_node(Node::default()),
        );
        builder.worklist.push(initial_states);
        while let Some(active) = builder.worklist.pop() {
            builder.closure(&active);
        }
        dbg!(builder.states);
        builder.dfa
    }

    pub fn exec(&self, input: &[u8]) -> Option<Vec<usize>> {
        let mut last = self.initial.to_vec();
        let mut next = Vec::new();
        let mut best = None;
        let mut state = NodeIndex::new(0);

        let mut apply_effects = |idx, effects: &[Effect]| {
            debug_assert!(next.is_empty());
            for effect in effects.iter().copied() {
                match effect {
                    Effect::CopyFrom(thread) => next.push(last[thread as usize].clone()),
                    Effect::SaveTo(register) => {
                        Rc::make_mut(next.last_mut().unwrap())[register as usize] = idx
                    }
                    Effect::Match => best = Some(next.last().unwrap().clone()),
                }
            }
            last.clear();
            std::mem::swap(&mut last, &mut next);
        };

        'next: for (idx, &b) in input.iter().enumerate() {
            for edge in self.graph.edges(state) {
                let target = edge.target();
                let edge = edge.weight();
                if edge.range_lo <= b && b <= edge.range_hi {
                    apply_effects(idx + 1, &edge.effects);
                    state = target;
                    continue 'next;
                }
            }

            // The state machine is stuck; return early with the best match found.
            return best.map(|rc| rc.to_vec());
        }

        // Check for a final transition on end-of-input.
        for edge in self.graph.edges(state) {
            let edge = edge.weight();
            if edge.range_hi < edge.range_lo {
                apply_effects(input.len(), &edge.effects);
                break;
            }
        }
        best.map(|rc| rc.to_vec())
    }

    pub fn to_dot(&self) -> String {
        use std::fmt::Write;

        let dot = dot::Dot::with_attr_getters(
            &self.graph,
            &[
                dot::Config::NodeIndexLabel,
                dot::Config::EdgeNoLabel,
                dot::Config::GraphContentOnly,
            ],
            &|_dfa, edge| {
                let edge = edge.weight();
                let mut result = "label=\"".to_string();
                if edge.range_lo > edge.range_hi {
                    result.push('$');
                } else if edge.range_lo == edge.range_hi {
                    write!(&mut result, "{:?}", char::from(edge.range_lo)).unwrap();
                } else {
                    write!(
                        &mut result,
                        "[{:?}-{:?}]",
                        char::from(edge.range_lo),
                        char::from(edge.range_hi)
                    )
                    .unwrap();
                }
                let mut new_thread = 0;
                for effect in edge.effects.iter() {
                    match effect {
                        Effect::CopyFrom(old_thread) => {
                            write!(&mut result, "\\n{new_thread}: copy {old_thread}").unwrap();
                            new_thread += 1;
                        }
                        Effect::SaveTo(reg) => write!(&mut result, ", set {reg}").unwrap(),
                        Effect::Match => write!(&mut result, ", match").unwrap(),
                    }
                }
                result.push('"');
                result
            },
            &|_dfa, _node| String::new(),
        );

        let mut initial = String::new();
        for (thread, regs) in self.initial.iter().enumerate() {
            write!(&mut initial, "{thread}:").unwrap();
            for &reg in regs.iter() {
                if reg == usize::MAX {
                    initial.push_str(" ?");
                } else {
                    write!(&mut initial, " {reg}").unwrap();
                }
            }
            initial.push_str("\\n");
        }

        format!("digraph {{\n    0 [ xlabel=\"{initial}\" ]\n{dot:?}\n}}")
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Node;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Effect {
    CopyFrom(u16),
    SaveTo(u16),
    Match,
}

#[derive(Clone, Debug)]
struct Edge {
    range_lo: u8,
    range_hi: u8,
    effects: Box<[Effect]>,
}

#[derive(Clone, Debug, Default)]
struct State {
    active: Vec<(u16, Constraint)>,
    effects: Vec<Effect>,
}

struct Builder<'a> {
    program: &'a Program,
    states: HashMap<Box<[(u16, Constraint)]>, NodeIndex>,
    worklist: Vec<Box<[(u16, Constraint)]>>,
    dfa: DFA,
    active: Vec<FlagSet<Constraint>>,
    current_effects: Vec<Effect>,
    current_edges: BTreeMap<u8, Option<State>>,
}

impl<'a> Builder<'a> {
    fn closure(&mut self, starts: &[(u16, Constraint)]) {
        debug_assert!(self.current_edges.is_empty());
        self.current_edges.insert(0, None);

        let mut current_end = None;

        for (idx, &(pc, constraint)) in starts.iter().enumerate() {
            debug_assert!(self.current_effects.is_empty());
            self.current_effects
                .push(Effect::CopyFrom(idx.try_into().unwrap()));
            match self.program.buf[usize::from(pc)] {
                Inst::Range(lo, hi) => {
                    self.active.fill(FlagSet::default());
                    self.gather(pc + 1, lo, hi, constraint, Constraint::Anything);
                }
                Inst::Match => {
                    self.current_effects.push(Effect::Match);
                    self.add_state(0, 255, constraint, None);
                    if constraint.match_at_end() && current_end.is_none() {
                        current_end = Some(State {
                            active: Vec::new(),
                            effects: self.current_effects.clone(),
                        });
                    }
                    self.current_effects.pop();
                }
                Inst::Assertion(_) | Inst::Save(_) | Inst::Jmp(_) | Inst::Split { .. } => {
                    unreachable!();
                }
            }
            self.current_effects.pop();
        }

        let from = self.states[starts];
        if let Some(state) = current_end {
            self.add_edge(from, state, 255, 0);
        }

        let current_edges = std::mem::take(&mut self.current_edges);
        let mut edges = current_edges.into_iter().peekable();
        while let Some((lo, state)) = edges.next() {
            if let Some(state) = state {
                let hi = edges.peek().map_or(u8::MAX, |(next, _)| next - 1);
                self.add_edge(from, state, lo, hi);
            }
        }
    }

    fn gather(&mut self, pc: u16, lo: u8, hi: u8, last: Constraint, next: Constraint) {
        let active = &mut self.active[usize::from(pc)];
        if !active.is_disjoint(next) {
            return;
        }
        *active |= next;

        match self.program.buf[usize::from(pc)] {
            Inst::Range(..) | Inst::Match => self.add_state(lo, hi, last, Some((pc, next))),
            Inst::Assertion(kind) => {
                match kind {
                    // After matching a byte, StartText can't match; prune here.
                    Assertions::StartText => {}
                    // After matching a byte, StartLine can only match if it was the end of a line.
                    Assertions::StartLine => {
                        if let Some(last) = last.intersection(Constraint::Newline) {
                            if last.overlaps(lo, hi) {
                                self.gather(pc + 1, lo, hi, last, next);
                            }
                        }
                    }
                    // These assertions apply to the following byte, not the preceding one.
                    Assertions::EndLine => {
                        if let Some(next) = next.intersection(Constraint::Newline) {
                            self.gather(pc + 1, lo, hi, last, next);
                        }
                    }
                    Assertions::EndText => {
                        if let Some(next) = next.intersection(Constraint::EndText) {
                            self.gather(pc + 1, lo, hi, last, next);
                        }
                    }
                    // Word boundaries need the preceding byte range split into word/non-word.
                    Assertions::AsciiWordBoundary => {
                        if let Some(last) = last.intersection(Constraint::WordByte) {
                            if last.overlaps(lo, hi) {
                                if let Some(next) = next.intersection(Constraint::NotWordByte) {
                                    self.gather(pc + 1, lo, hi, last, next);
                                }
                            }
                        }
                        if let Some(last) = last.intersection(Constraint::NotWordByte) {
                            if last.overlaps(lo, hi) {
                                if let Some(next) = next.intersection(Constraint::WordByte) {
                                    self.gather(pc + 1, lo, hi, last, next);
                                }
                            }
                        }
                    }
                    Assertions::AsciiNotWordBoundary => {
                        if let Some(last) = last.intersection(Constraint::WordByte) {
                            if last.overlaps(lo, hi) {
                                if let Some(next) = next.intersection(Constraint::WordByte) {
                                    self.gather(pc + 1, lo, hi, last, next);
                                }
                            }
                        }
                        if let Some(last) = last.intersection(Constraint::NotWordByte) {
                            if last.overlaps(lo, hi) {
                                if let Some(next) = next.intersection(Constraint::NotWordByte) {
                                    self.gather(pc + 1, lo, hi, last, next);
                                }
                            }
                        }
                    }
                }
            }
            Inst::Save(reg) => {
                let effect = Effect::SaveTo(reg);
                if self.current_effects.contains(&effect) {
                    self.gather(pc + 1, lo, hi, last, next);
                } else {
                    self.current_effects.push(effect);
                    self.gather(pc + 1, lo, hi, last, next);
                    self.current_effects.pop();
                }
            }
            Inst::Jmp(pc) => self.gather(pc, lo, hi, last, next),
            Inst::Split { prefer_next, to } => {
                let (a, b) = if prefer_next {
                    (pc + 1, to)
                } else {
                    (to, pc + 1)
                };
                self.gather(a, lo, hi, last, next);
                self.gather(b, lo, hi, last, next);
            }
        }
    }

    fn add_state(&mut self, lo: u8, hi: u8, last: Constraint, next: Option<(u16, Constraint)>) {
        for (lo, hi) in last.ranges(lo, hi) {
            self.split(lo);
            if let Some(next) = hi.checked_add(1) {
                self.split(next);
            }
            for (_, state) in self.current_edges.range_mut(lo..=hi) {
                let state = state.get_or_insert_with(State::default);
                if !matches!(state.effects.last(), Some(Effect::Match)) {
                    if let Some(next) = next {
                        if state.active.contains(&next) {
                            continue;
                        }
                        state.active.push(next);
                    }
                    state.effects.extend_from_slice(&self.current_effects);
                }
            }
        }
    }

    fn split(&mut self, at: u8) {
        let (k, v) = self.current_edges.range(..=at).rev().next().unwrap();
        if *k != at {
            let v = v.clone();
            self.current_edges.insert(at, v);
        }
    }

    fn add_edge(&mut self, from: NodeIndex, state: State, range_lo: u8, range_hi: u8) {
        let to = self
            .states
            .entry(state.active.as_slice().into())
            .or_insert_with(|| {
                self.worklist.push(state.active.into_boxed_slice());
                self.dfa.graph.add_node(Node::default())
            });
        self.dfa.graph.add_edge(
            from,
            *to,
            Edge {
                range_lo,
                range_hi,
                effects: state.effects.into_boxed_slice(),
            },
        );
    }
}
