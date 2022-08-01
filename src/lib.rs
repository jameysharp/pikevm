use bitvec::vec::BitVec;
use flagset::{flags, FlagSet};
use regex_syntax::hir::{self, Hir, HirKind};
use regex_syntax::utf8::Utf8Sequences;
use regex_syntax::{is_word_byte, ParserBuilder};
use std::borrow::Borrow;
use std::rc::Rc;

pub mod dfa;

pub fn compile(pat: &str) -> regex_syntax::Result<Program> {
    let mut result = Program {
        buf: Vec::new(),
        registers: 2,
    };

    let mut parse_config = ParserBuilder::new();
    parse_config.allow_invalid_utf8(true);
    parse_config.unicode(false);

    let hir = parse_config.build().parse(pat)?;

    if !hir.is_anchored_start() {
        result.compile_hir(&Hir::repetition(hir::Repetition {
            kind: hir::RepetitionKind::ZeroOrMore,
            greedy: false,
            hir: Box::new(Hir::any(true)),
        }));
    }

    result.push(Inst::Save(0));
    result.compile_hir(&hir);
    result.push(Inst::Save(1));
    result.push(Inst::Match);

    dbg!(&result);
    Ok(result)
}

flags! {
    enum Assertions: u8 {
        StartLine,
        EndLine,
        StartText,
        EndText,
        AsciiWordBoundary,
        AsciiNotWordBoundary,
    }
}

#[derive(Clone, Debug)]
enum Inst {
    Range(u8, u8),
    Assertion(Assertions),
    Match,
    Save(u16),
    Jmp(u16),
    Split { prefer_next: bool, to: u16 },
}

#[derive(Clone, Debug)]
pub struct Program {
    buf: Vec<Inst>,
    registers: u16,
}

impl Program {
    fn loc(&self) -> u16 {
        self.buf.len() as u16
    }

    fn push(&mut self, inst: Inst) -> u16 {
        let loc = self.loc();
        self.buf.push(inst);
        loc
    }

    fn placeholder(&mut self) -> u16 {
        self.push(Inst::Jmp(u16::MAX))
    }

    fn patch(&mut self, loc: u16, inst: Inst) {
        self.buf[loc as usize] = inst;
    }

    fn alts<T, F: FnMut(&mut Self, T)>(&mut self, alts: impl IntoIterator<Item = T>, mut f: F) {
        let prefer_next = true; // prefer left-most successful alternative
        let mut iter = alts.into_iter();
        let mut last = iter.next().unwrap();
        let mut jmps = Vec::new();
        for next in iter {
            let split = self.placeholder();
            f(self, last);
            jmps.push(self.placeholder());
            let to = self.loc();
            self.patch(split, Inst::Split { prefer_next, to });
            last = next;
        }
        f(self, last);
        let to = self.loc();
        for jmp in jmps {
            self.patch(jmp, Inst::Jmp(to));
        }
    }

    fn compile_hir(&mut self, hir: &Hir) -> bool {
        match hir.kind() {
            HirKind::Empty => true,
            HirKind::Literal(hir::Literal::Unicode(u)) => {
                let mut buf = [0; 4];
                for b in u.encode_utf8(&mut buf).bytes() {
                    self.push(Inst::Range(b, b));
                }
                false
            }
            &HirKind::Literal(hir::Literal::Byte(b)) => {
                self.push(Inst::Range(b, b));
                false
            }
            HirKind::Class(hir::Class::Unicode(uc)) => {
                let seqs = uc
                    .iter()
                    .flat_map(|range| Utf8Sequences::new(range.start(), range.end()));
                self.alts(seqs, |p, seq| {
                    for byte_range in seq.as_slice() {
                        p.push(Inst::Range(byte_range.start, byte_range.end));
                    }
                });
                false
            }
            HirKind::Class(hir::Class::Bytes(bc)) => {
                self.alts(bc.iter(), |p, range| {
                    p.push(Inst::Range(range.start(), range.end()));
                });
                false
            }
            HirKind::Anchor(kind) => {
                self.push(Inst::Assertion(match kind {
                    hir::Anchor::StartLine => Assertions::StartLine,
                    hir::Anchor::EndLine => Assertions::EndLine,
                    hir::Anchor::StartText => Assertions::StartText,
                    hir::Anchor::EndText => Assertions::EndText,
                }));
                true
            }
            HirKind::WordBoundary(kind) => {
                self.push(Inst::Assertion(match kind {
                    hir::WordBoundary::Ascii => Assertions::AsciiWordBoundary,
                    hir::WordBoundary::AsciiNegate => Assertions::AsciiNotWordBoundary,
                    _ => unimplemented!(),
                }));
                true
            }
            HirKind::Repetition(rep) => match rep.kind {
                hir::RepetitionKind::ZeroOrOne => {
                    let split = self.placeholder();
                    let nullable = self.compile_hir(&*rep.hir);
                    let to = self.loc();
                    let prefer_next = rep.greedy;
                    self.patch(split, Inst::Split { prefer_next, to });
                    if nullable {
                        self.nullable_captures(&*rep.hir);
                    }
                    true
                }
                hir::RepetitionKind::ZeroOrMore => {
                    let split = self.placeholder();
                    let nullable = self.compile_hir(&*rep.hir);
                    self.push(Inst::Jmp(split));
                    let to = self.loc();
                    let prefer_next = rep.greedy;
                    self.patch(split, Inst::Split { prefer_next, to });
                    if nullable {
                        self.nullable_captures(&*rep.hir);
                    }
                    true
                }
                hir::RepetitionKind::OneOrMore => {
                    let to = self.loc();
                    let nullable = self.compile_hir(&*rep.hir);
                    let prefer_next = !rep.greedy;
                    self.push(Inst::Split { prefer_next, to });
                    if nullable {
                        self.nullable_captures(&*rep.hir);
                    }
                    nullable
                }
                hir::RepetitionKind::Range(_) => unimplemented!(),
            },
            HirKind::Group(group) => match group.kind {
                hir::GroupKind::CaptureIndex(index) | hir::GroupKind::CaptureName { index, .. } => {
                    let register = (index * 2) as _;
                    self.registers = register + 2;
                    self.push(Inst::Save(register));
                    let nullable = self.compile_hir(&*group.hir);
                    self.push(Inst::Save(register + 1));
                    nullable
                }
                hir::GroupKind::NonCapturing => self.compile_hir(&*group.hir),
            },
            HirKind::Concat(subs) => {
                let mut nullable = true;
                for hir in subs {
                    nullable &= self.compile_hir(hir);
                }
                nullable
            }
            HirKind::Alternation(subs) => {
                let mut nullable = false;
                self.alts(subs, |p, hir| {
                    nullable |= p.compile_hir(hir);
                });
                nullable
            }
        }
    }

    /// Fix up match positions to match PCRE's weird behavior on repetitions of subexpressions
    /// which can match the empty string. Any nested groups that would capture when the whole
    /// subexpression is given the empty string should always capture after the repetition ends.
    fn nullable_captures(&mut self, hir: &Hir) -> bool {
        let undo = self.buf.len();
        let mut try_sub = |hir| {
            let nullable = self.nullable_captures(hir);
            if !nullable {
                self.buf.truncate(undo);
            }
            nullable
        };
        match hir.kind() {
            HirKind::Empty => true,
            HirKind::Literal(_) => false,
            HirKind::Class(_) => false,
            HirKind::Anchor(_) => true,
            HirKind::WordBoundary(_) => true,
            HirKind::Repetition(rep) => {
                let nullable = try_sub(&*rep.hir);
                match rep.kind {
                    hir::RepetitionKind::ZeroOrOne | hir::RepetitionKind::ZeroOrMore => true,
                    hir::RepetitionKind::OneOrMore => nullable,
                    hir::RepetitionKind::Range(_) => unimplemented!(),
                }
            }
            HirKind::Group(group) => {
                let nullable = try_sub(&*group.hir);
                match group.kind {
                    hir::GroupKind::CaptureIndex(index)
                    | hir::GroupKind::CaptureName { index, .. } => {
                        if nullable {
                            let register = (index * 2) as _;
                            self.push(Inst::Save(register));
                            self.push(Inst::Save(register + 1));
                        }
                    }
                    hir::GroupKind::NonCapturing => {}
                }
                nullable
            }
            HirKind::Concat(subs) => subs.iter().all(try_sub),
            HirKind::Alternation(subs) => subs.iter().any(try_sub),
        }
    }

    pub fn exec(&self, input: &[u8]) -> Option<Vec<usize>> {
        exec_many(input, &[self]).into_iter().next().unwrap()
    }

    pub fn to_dfa(&self) -> dfa::DFA {
        dfa::DFA::new(self)
    }
}

pub fn exec_many(input: &[u8], patterns: &[impl Borrow<Program>]) -> Vec<Option<Vec<usize>>> {
    let mut patterns = patterns
        .iter()
        .map(|pattern| (Threads::new(pattern.borrow()), None))
        .collect::<Vec<_>>();
    let mut current = Vec::new();

    let mut last_sp = None;
    for (idx, sp) in input.iter().copied().enumerate() {
        dbg!(idx, sp);

        let mut progress = false;
        for (threads, result) in patterns.iter_mut() {
            threads.take(&mut current);
            dbg!(&current);
            if current.is_empty() {
                continue;
            }

            progress = true;
            for mut thread in current.drain(..) {
                dbg!(thread.pc, &threads.program.buf[thread.pc as usize]);

                if !thread.check_assertions(last_sp, Some(sp)) {
                    continue;
                }

                match threads.program.buf[thread.pc as usize] {
                    Inst::Range(lo, hi) => {
                        if sp >= lo && sp <= hi {
                            threads.add(idx + 1, thread.next());
                        }
                    }
                    Inst::Match => {
                        *result = Some(thread.saved);
                        dbg!(result);
                        break;
                    }
                    Inst::Assertion(_) | Inst::Save(_) | Inst::Jmp(_) | Inst::Split { .. } => {
                        unreachable!()
                    }
                }
            }
        }

        if !progress {
            break;
        }

        last_sp = Some(sp);
    }

    debug_assert!(current.is_empty());
    drop(current);

    patterns
        .into_iter()
        .map(|(threads, mut result)| {
            let Threads { list: current, .. } = threads;
            dbg!(&current);
            for mut thread in current {
                if let Inst::Match = threads.program.buf[thread.pc as usize] {
                    if thread.check_assertions(last_sp, None) {
                        result = Some(thread.saved);
                        dbg!(&result);
                        break;
                    }
                }
            }
            result.map(|rc| rc.to_vec())
        })
        .collect()
}

#[derive(Clone, Debug)]
struct Thread {
    pc: u16,
    saved: Rc<Vec<usize>>,
    assertions: FlagSet<Assertions>,
}

impl Thread {
    fn new(pc: u16, saved: Rc<Vec<usize>>) -> Self {
        Thread {
            pc,
            saved,
            assertions: FlagSet::default(),
        }
    }

    fn next(mut self) -> Self {
        self.pc += 1;
        self
    }

    fn check_assertions(&mut self, prev: Option<u8>, next: Option<u8>) -> bool {
        for assertion in self.assertions.drain() {
            let passed = match assertion {
                Assertions::StartText => prev.is_none(),
                Assertions::EndText => next.is_none(),
                Assertions::StartLine => prev.unwrap_or(b'\n') == b'\n',
                Assertions::EndLine => next.unwrap_or(b'\n') == b'\n',
                Assertions::AsciiWordBoundary => {
                    prev.map_or(false, is_word_byte) != next.map_or(false, is_word_byte)
                }
                Assertions::AsciiNotWordBoundary => {
                    prev.map_or(false, is_word_byte) == next.map_or(false, is_word_byte)
                }
            };
            if !passed {
                return false;
            }
        }
        true
    }
}

struct Threads<'a> {
    program: &'a Program,
    active: BitVec,
    list: Vec<Thread>,
}

impl<'a> Threads<'a> {
    fn new(program: &Program) -> Threads {
        let mut saved = Vec::new();
        saved.resize(program.registers as usize, usize::MAX);
        let mut result = Threads {
            program,
            active: BitVec::repeat(false, program.buf.len()),
            list: Vec::new(),
        };
        result.add(0, Thread::new(0, Rc::new(saved)));
        result
    }

    fn take(&mut self, buf: &mut Vec<Thread>) {
        std::mem::swap(&mut self.list, buf);
        self.active.fill(false);
    }

    fn add(&mut self, idx: usize, mut thread: Thread) {
        if self.active.replace(thread.pc as usize, true) {
            return;
        }
        // NFA epsilon closure: a thread can only stop on Range or Match instructions. For anything
        // else, recurse on the targets of the instruction. Note that this recursion is at worst
        // O(n) in the number of instructions; any epsilon cycles are broken using self.active.
        match self.program.buf[thread.pc as usize] {
            Inst::Range(_, _) | Inst::Match => {
                self.list.push(thread);
            }
            Inst::Assertion(kind) => {
                thread.assertions |= kind;
                self.add(idx, thread.next());
            }
            Inst::Save(reg) => {
                Rc::make_mut(&mut thread.saved)[reg as usize] = idx;
                self.add(idx, thread.next());
            }
            Inst::Jmp(pc) => self.add(idx, Thread { pc, ..thread }),
            Inst::Split { prefer_next, to } => {
                let (a, b) = if prefer_next {
                    (thread.pc + 1, to)
                } else {
                    (to, thread.pc + 1)
                };
                self.add(
                    idx,
                    Thread {
                        pc: a,
                        saved: thread.saved.clone(),
                        ..thread
                    },
                );
                self.add(idx, Thread { pc: b, ..thread });
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn pathological() {
        let p = compile("^a*a*a*a*a*a*a*a*a*b$").unwrap();
        let input = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabc";
        assert_eq!(p.exec(input), None);
    }
}
