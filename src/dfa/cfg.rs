use bitvec::vec::BitVec;
use log::trace;
use petgraph::algo::dominators::simple_fast;
use petgraph::dot;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use super::{Effect, DFA};

#[derive(Clone, Debug, Default)]
pub struct CFG {
    graph: Graph<Block, Branch>,
}

impl CFG {
    pub fn exec(&self, input: &[u8]) -> Option<Vec<usize>> {
        let mut register = HashMap::new();
        let mut current_block = NodeIndex::new(0);
        let mut current_args = Vec::new();

        register.insert(Value::UNSET, usize::MAX);
        register.insert(Value::START, 0);

        loop {
            let block = &self.graph[current_block];
            debug_assert_eq!(block.params.len(), current_args.len());
            for (value, &param) in current_args.drain(..).zip(block.params.iter()) {
                register.insert(param, value);
                trace!("{param} := {value}");
            }

            for op in block.program.iter() {
                trace!("eval {op:?}");
                let value = register[&op.operand].wrapping_add(op.constant as usize);
                trace!("{} := {}", op.result, value);
                register.insert(op.result, value);
            }

            let current_edge;
            let mut edges = self.graph.edges(current_block);
            match &block.terminator {
                Terminator::Goto => {
                    current_edge = edges.next().unwrap();
                    debug_assert!(edges.next().is_none());
                }
                Terminator::InputAt(reg) => {
                    let offset = register[reg];
                    let edge = if let Some(&b) = input.get(offset) {
                        edges.find(|edge| edge.weight().contains(b))
                    } else {
                        edges.find(|edge| edge.weight().contains_eof())
                    };
                    current_edge = edge.unwrap_or_else(|| {
                        self.graph
                            .edges(current_block)
                            .find(|edge| edge.weight().contains_default())
                            .unwrap()
                    });
                }
                Terminator::Return(captures) => {
                    let captures: Vec<_> = captures.iter().map(|reg| register[reg]).collect();
                    return Some(captures).filter(|captures| captures[1] != usize::MAX);
                }
            }

            current_block = current_edge.target();
            trace!(
                "taking edge {} to {}: {:?}",
                current_edge.id().index(),
                current_block.index(),
                current_edge.weight(),
            );
            current_args.extend(current_edge.weight().args.iter().map(|arg| register[arg]));
        }
    }

    pub fn to_dot(&self) -> impl std::fmt::Debug + '_ {
        dot::Dot::new(&self.graph)
    }
}

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Value(u32);

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl Value {
    const UNSET: Value = Value(0);
    const START: Value = Value(1);
    const FIRST: Value = Value(2);

    fn next(&mut self) -> Self {
        let next = *self;
        self.0 += 1;
        next
    }
}

#[derive(Clone)]
struct Add {
    result: Value,
    operand: Value,
    constant: isize,
}

impl std::fmt::Debug for Add {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} := {} + {}", self.result, self.operand, self.constant)
    }
}

#[derive(Clone, Default)]
enum Terminator {
    #[default]
    Goto,
    InputAt(Value),
    Return(Box<[Value]>),
}

impl std::fmt::Debug for Terminator {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Terminator::Goto => f.write_str("goto"),
            &Terminator::InputAt(reg) => write!(f, "switch(input[{}])", reg),
            Terminator::Return(captures) => write!(f, "return {:?}", captures),
        }
    }
}

impl Terminator {
    fn registers(&self) -> &[Value] {
        match self {
            Terminator::Goto => &[],
            Terminator::InputAt(reg) => std::slice::from_ref(reg),
            Terminator::Return(captures) => captures,
        }
    }

    fn registers_mut(&mut self) -> &mut [Value] {
        match self {
            Terminator::Goto => &mut [],
            Terminator::InputAt(reg) => std::slice::from_mut(reg),
            Terminator::Return(captures) => captures,
        }
    }
}

#[derive(Clone, Default)]
struct Block {
    params: Vec<Value>,
    program: Vec<Add>,
    terminator: Terminator,
}

impl std::fmt::Debug for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "{:?}:", self.params)?;
        for op in self.program.iter() {
            writeln!(f, "{:?}", op)?;
        }
        write!(f, "{:?}", self.terminator)
    }
}

#[derive(Clone)]
struct Branch {
    range_lo: u8,
    range_hi: u8,
    args: Vec<Value>,
}

impl std::fmt::Debug for Branch {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.contains_eof() {
            f.write_str("$")?;
        } else if self.contains_default() {
            f.write_str("default")?;
        } else if self.range_lo == self.range_hi {
            write!(f, "{:?}", char::from(self.range_lo))?;
        } else {
            write!(
                f,
                "[{:?}-{:?}]",
                char::from(self.range_lo),
                char::from(self.range_hi)
            )?;
        }
        write!(f, " => {:?}", self.args)
    }
}

impl Branch {
    fn contains(&self, b: u8) -> bool {
        self.range_lo <= b && b <= self.range_hi
    }

    fn contains_eof(&self) -> bool {
        self.range_lo == u8::MAX && self.range_hi == 0
    }

    fn contains_default(&self) -> bool {
        self.range_lo == u8::MAX && self.range_hi == 1
    }
}

impl From<&DFA> for CFG {
    fn from(dfa: &DFA) -> CFG {
        let mut cfg = CFG::default();
        let mut counter = Value::FIRST;
        let registers_per_set = dfa.initial[0].len();

        let mut pending_edges = Vec::new();
        let entry_block = cfg.graph.add_node(Block::default());

        let params: Vec<_> = (0..registers_per_set).map(|_| counter.next()).collect();
        let exit_block = cfg.graph.add_node(Block {
            params: params.clone(),
            program: Vec::new(),
            terminator: Terminator::Return(params.into_boxed_slice()),
        });

        let mut block_for_state = HashMap::new();
        for state in dfa.graph.node_indices() {
            let sets =
                if let Some(edge) = dfa.graph.edges_directed(state, Direction::Incoming).next() {
                    let effects = &edge.weight().effects[..];
                    let mut sets = effects
                        .iter()
                        .filter(|eff| matches!(eff, Effect::CopyFrom(_)))
                        .count();
                    if !matches!(effects.last(), Some(Effect::Match)) {
                        sets += 1;
                    }
                    sets
                } else {
                    debug_assert_eq!(state, NodeIndex::new(0));
                    dfa.initial.len() + 1
                };

            let params: Vec<_> = (0..sets * registers_per_set + 1)
                .map(|_| counter.next())
                .collect();
            let now = params[0];
            let next = counter.next();
            let block = cfg.graph.add_node(Block {
                params: params.clone(),
                program: vec![Add {
                    result: next,
                    operand: now,
                    constant: 1,
                }],
                terminator: Terminator::InputAt(now),
            });
            block_for_state.insert(state, block);

            for edge in dfa.graph.edges(state) {
                let branch = edge.weight();
                let mut args = Vec::new();
                args.push(next);
                for &effect in branch.effects.iter() {
                    match effect {
                        Effect::CopyFrom(set) => {
                            let set = &params[1 + usize::from(set) * registers_per_set..]
                                [..registers_per_set];
                            args.extend_from_slice(set);
                        }
                        Effect::SaveTo(reg) => {
                            let last = args.len() - registers_per_set;
                            let set = &mut args[last..];
                            set[usize::from(reg)] = next;
                        }
                        Effect::Match => {}
                    }
                }

                if !matches!(branch.effects.last(), Some(Effect::Match)) {
                    // This branch didn't overwrite the previous best match, so inherit that.
                    let set = &params[params.len() - registers_per_set..];
                    args.extend_from_slice(set);
                }

                pending_edges.push((
                    block,
                    edge.target(),
                    Branch {
                        range_lo: branch.range_lo,
                        range_hi: branch.range_hi,
                        args,
                    },
                ));
            }

            let set = &params[params.len() - registers_per_set..];
            cfg.graph.add_edge(
                block,
                exit_block,
                Branch {
                    range_lo: u8::MAX,
                    range_hi: 1,
                    args: set.into(),
                },
            );
        }

        let mut args = Vec::new();
        args.push(Value::START);
        args.extend(
            dfa.initial
                .iter()
                .flat_map(|set| &set[..])
                .map(|&v| match v {
                    0 => Value::START,
                    usize::MAX => Value::UNSET,
                    _ => unreachable!(),
                }),
        );
        args.extend((0..registers_per_set).map(|_| Value::UNSET));

        cfg.graph.add_edge(
            entry_block,
            block_for_state[&NodeIndex::new(0)],
            Branch {
                range_lo: u8::MAX,
                range_hi: 1,
                args,
            },
        );

        for (block, state, branch) in pending_edges {
            cfg.graph.add_edge(block, block_for_state[&state], branch);
        }

        let defined_in = defined_in(&cfg);

        // The order of these optimization passes matters.
        // - merge_equivalent_args must precede copy_propagation: it currently assumes that all
        //   registers used within a block are also defined in that block (usually as a block
        //   parameter), but copy_propagation breaks that assumption.
        // - copy_propagation must precede dead_code_elimination: it leaves unused values behind.
        merge_equivalent_args(&mut cfg);
        copy_propagation(&mut cfg, &defined_in);
        dead_code_elimination(&mut cfg, &defined_in);
        cfg
    }
}

fn defined_in(cfg: &CFG) -> HashMap<Value, NodeIndex> {
    let mut defined_in = HashMap::new();
    defined_in.insert(Value::UNSET, NodeIndex::new(0));
    defined_in.insert(Value::START, NodeIndex::new(0));
    for node in cfg.graph.node_indices() {
        let block = &cfg.graph[node];
        for &reg in block.params.iter() {
            let dup = defined_in.insert(reg, node);
            debug_assert!(dup.is_none());
        }
        for op in block.program.iter() {
            let dup = defined_in.insert(op.result, node);
            debug_assert!(dup.is_none());
        }
    }
    defined_in
}

#[derive(Debug, Default)]
struct Worklist<T> {
    todo: Vec<T>,
    pending: HashSet<T>,
}

impl<T: Copy + Eq + std::hash::Hash> Worklist<T> {
    fn push(&mut self, todo: T) {
        if self.pending.insert(todo) {
            self.todo.push(todo);
        }
    }

    fn pop(&mut self) -> Option<T> {
        self.todo.pop().filter(|v| self.pending.remove(v))
    }
}

fn merge_equivalent_args(cfg: &mut CFG) {
    let mut equivs: Vec<Vec<usize>> = vec![Vec::new(); cfg.graph.node_count()];
    let mut worklist = Worklist::default();
    worklist.push(NodeIndex::new(0));

    let mut incoming = HashMap::new();
    let mut first_use = HashMap::new();
    let mut agree_at = HashMap::new();
    while let Some(node) = worklist.pop() {
        let p = &cfg.graph[node].params;
        incoming.clear();
        incoming.extend(
            p.iter()
                .zip(equivs[node.index()].iter())
                .map(|(&param, &repr)| (param, p[repr])),
        );

        for edge in cfg.graph.edges(node) {
            first_use.clear();
            let local = edge.weight().args.iter().enumerate().map(|(idx, arg)| {
                *first_use
                    .entry(*incoming.get(arg).unwrap_or(arg))
                    .or_insert(idx)
            });

            let global = &mut equivs[edge.target().index()];
            let mut changed = false;
            if global.is_empty() {
                global.extend(local);
                changed = !global.is_empty();
            } else {
                agree_at.clear();
                debug_assert_eq!(global.len(), edge.weight().args.len());
                for (idx, (g, l)) in global.iter_mut().zip(local).enumerate() {
                    if *g != l {
                        let at = *agree_at.entry((*g, l)).or_insert(idx);
                        if *g != at {
                            *g = at;
                            changed = true;
                        }
                    }
                }
            }

            if changed {
                worklist.push(edge.target());
            }
        }
    }

    trace!("equivalent args: {:?}", equivs);

    let mut replacements = HashMap::new();
    for (equiv, block) in equivs.iter().zip(cfg.graph.node_weights_mut()) {
        let local = block
            .params
            .iter()
            .zip(equiv.iter().map(|&repr| block.params[repr]));
        for (&orig, repr) in local {
            if orig != repr {
                let dup = replacements.insert(orig, repr);
                debug_assert!(dup.is_none());
            }
        }

        for op in block.program.iter_mut() {
            if let Some(&replacement) = replacements.get(&op.operand) {
                op.operand = replacement;
            }
        }

        for reg in block.terminator.registers_mut() {
            if let Some(&replacement) = replacements.get(reg) {
                *reg = replacement;
            }
        }

        let mut keep = equiv.iter().enumerate().map(|(idx, &repr)| idx == repr);
        block.params.retain(|_| keep.next().unwrap());
        debug_assert!(keep.next().is_none());
    }

    for edge in cfg.graph.edge_indices() {
        let (_, target) = cfg.graph.edge_endpoints(edge).unwrap();
        let mut keep = equivs[target.index()]
            .iter()
            .enumerate()
            .map(|(idx, &repr)| idx == repr);
        cfg.graph[edge].args.retain_mut(|v| {
            if let Some(&replacement) = replacements.get(v) {
                *v = replacement;
            }
            keep.next().unwrap()
        });
        debug_assert!(keep.next().is_none());
    }
}

fn copy_propagation(cfg: &mut CFG, defined_in: &HashMap<Value, NodeIndex>) {
    let dominators = simple_fast(&cfg.graph, NodeIndex::new(0));

    let mut worklist = Worklist::default();
    worklist.push(NodeIndex::new(0));

    let mut equivs = HashMap::new();
    while let Some(node) = worklist.pop() {
        for edge in cfg.graph.edges(node) {
            let args = &edge.weight().args;
            let params = &cfg.graph[edge.target()].params;
            for (&arg, &param) in args.iter().zip(params.iter()) {
                let arg = equivs.get(&arg).copied().flatten().unwrap_or(arg);
                let repr = Some(arg).filter(|arg| {
                    let arg_def = defined_in[arg];
                    dominators
                        .strict_dominators(edge.target())
                        .unwrap()
                        .any(|n| n == arg_def)
                });

                match equivs.entry(param) {
                    Entry::Occupied(mut o) => {
                        if o.get().is_some() && *o.get() != repr {
                            o.insert(None);
                            worklist.push(edge.target());
                        }
                    }
                    Entry::Vacant(v) => {
                        v.insert(repr);
                        if repr.is_some() {
                            worklist.push(edge.target());
                        }
                    }
                }
            }
        }
    }

    trace!("equivalent copies: {:?}", equivs);

    for node in cfg.graph.node_indices() {
        let block = &mut cfg.graph[node];
        for op in block.program.iter_mut() {
            if let Some(&Some(repr)) = equivs.get(&op.operand) {
                op.operand = repr;
            }
        }
        for reg in block.terminator.registers_mut() {
            if let Some(&Some(repr)) = equivs.get(reg) {
                *reg = repr;
            }
        }
    }

    for edge in cfg.graph.edge_indices() {
        for reg in cfg.graph[edge].args.iter_mut() {
            if let Some(&Some(repr)) = equivs.get(reg) {
                *reg = repr;
            }
        }
    }
}

fn dead_code_elimination(cfg: &mut CFG, defined_in: &HashMap<Value, NodeIndex>) {
    let mut live = HashSet::new();
    let mut worklist = Worklist::default();
    for node in cfg.graph.node_indices() {
        let block = &cfg.graph[node];
        for &reg in block.terminator.registers() {
            if live.insert(reg) {
                worklist.push(defined_in[&reg]);
            }
        }
    }

    while let Some(node) = worklist.pop() {
        let block = &cfg.graph[node];
        for op in block.program.iter() {
            if live.contains(&op.result) && live.insert(op.operand) {
                worklist.push(defined_in[&op.operand]);
            }
        }

        let live_params: BitVec = block
            .params
            .iter()
            .map(|param| live.contains(param))
            .collect();
        for incoming in cfg.graph.edges_directed(node, Direction::Incoming) {
            for (&reg, used) in incoming.weight().args.iter().zip(live_params.iter()) {
                if *used && live.insert(reg) {
                    worklist.push(defined_in[&reg]);
                }
            }
        }
    }

    trace!(
        "{} live of {} registers: {:?}",
        live.len(),
        live.iter().max().unwrap().0 + 1,
        &live
    );

    for node in cfg.graph.node_indices() {
        let block = &mut cfg.graph[node];
        let live_params: BitVec = block
            .params
            .iter()
            .map(|param| live.contains(param))
            .collect();
        block.params.retain(|reg| live.contains(reg));
        block.program.retain(|op| live.contains(&op.result));

        let mut next_edge = cfg.graph.first_edge(node, Direction::Incoming);
        while let Some(edge) = next_edge {
            next_edge = cfg.graph.next_edge(edge, Direction::Incoming);
            let mut live_params = live_params.iter();
            cfg.graph[edge]
                .args
                .retain(|_| *live_params.next().unwrap());
            debug_assert!(live_params.next().is_none());
        }
    }
}
