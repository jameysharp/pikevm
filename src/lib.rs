use bitvec::vec::BitVec;
use flagset::{flags, FlagSet};
use regex_syntax::hir::{self, Hir, HirKind};
use regex_syntax::utf8::Utf8Sequences;
use regex_syntax::{is_word_byte, ParserBuilder};
use std::borrow::Borrow;
use std::rc::Rc;

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

    fn alts<F: FnOnce(&mut Self)>(&mut self, alts: impl IntoIterator<Item = F>) {
        let prefer_next = true; // prefer left-most successful alternative
        let mut iter = alts.into_iter();
        let mut last = iter.next().unwrap();
        let mut jmps = Vec::new();
        for next in iter {
            let split = self.placeholder();
            last(self);
            jmps.push(self.placeholder());
            let to = self.loc();
            self.patch(split, Inst::Split { prefer_next, to });
            last = next;
        }
        last(self);
        let to = self.loc();
        for jmp in jmps {
            self.patch(jmp, Inst::Jmp(to));
        }
    }

    fn compile_hir(&mut self, hir: &Hir) {
        match hir.kind() {
            HirKind::Empty => {}
            HirKind::Literal(hir::Literal::Unicode(u)) => {
                let mut buf = [0; 4];
                for b in u.encode_utf8(&mut buf).bytes() {
                    self.push(Inst::Range(b, b));
                }
            }
            &HirKind::Literal(hir::Literal::Byte(b)) => {
                self.push(Inst::Range(b, b));
            }
            HirKind::Class(hir::Class::Unicode(uc)) => {
                self.alts(
                    uc.iter()
                        .flat_map(|range| Utf8Sequences::new(range.start(), range.end()))
                        .map(|seq| {
                            move |p: &mut Program| {
                                for byte_range in seq.as_slice() {
                                    p.push(Inst::Range(byte_range.start, byte_range.end));
                                }
                            }
                        }),
                );
            }
            HirKind::Class(hir::Class::Bytes(bc)) => {
                self.alts(bc.iter().map(|range| {
                    move |p: &mut Program| {
                        p.push(Inst::Range(range.start(), range.end()));
                    }
                }));
            }
            HirKind::Anchor(kind) => {
                self.push(Inst::Assertion(match kind {
                    hir::Anchor::StartLine => Assertions::StartLine,
                    hir::Anchor::EndLine => Assertions::EndLine,
                    hir::Anchor::StartText => Assertions::StartText,
                    hir::Anchor::EndText => Assertions::EndText,
                }));
            }
            HirKind::WordBoundary(kind) => {
                self.push(Inst::Assertion(match kind {
                    hir::WordBoundary::Ascii => Assertions::AsciiWordBoundary,
                    hir::WordBoundary::AsciiNegate => Assertions::AsciiNotWordBoundary,
                    _ => unimplemented!(),
                }));
            }
            HirKind::Repetition(rep) => match rep.kind {
                hir::RepetitionKind::ZeroOrOne => {
                    let split = self.placeholder();
                    self.compile_hir(&*rep.hir);
                    let to = self.loc();
                    let prefer_next = rep.greedy;
                    self.patch(split, Inst::Split { prefer_next, to });
                }
                hir::RepetitionKind::ZeroOrMore => {
                    let split = self.placeholder();
                    self.compile_hir(&*rep.hir);
                    self.push(Inst::Jmp(split));
                    let to = self.loc();
                    let prefer_next = rep.greedy;
                    self.patch(split, Inst::Split { prefer_next, to });
                }
                hir::RepetitionKind::OneOrMore => {
                    let to = self.loc();
                    self.compile_hir(&*rep.hir);
                    let prefer_next = !rep.greedy;
                    self.push(Inst::Split { prefer_next, to });
                }
                hir::RepetitionKind::Range(_) => unimplemented!(),
            },
            HirKind::Group(group) => {
                match group.kind {
                    hir::GroupKind::CaptureIndex(index)
                    | hir::GroupKind::CaptureName { index, .. } => {
                        let register = (index * 2) as _;
                        self.registers = register + 2;
                        self.push(Inst::Save(register));
                        self.compile_hir(&*group.hir);
                        self.push(Inst::Save(register + 1));
                    }
                    hir::GroupKind::NonCapturing => {
                        self.compile_hir(&*group.hir);
                    }
                };
            }
            HirKind::Concat(subs) => {
                for hir in subs {
                    self.compile_hir(hir);
                }
            }
            HirKind::Alternation(subs) => {
                self.alts(
                    subs.iter()
                        .map(|hir| move |p: &mut Program| p.compile_hir(hir)),
                );
            }
        }
    }

    pub fn exec(&self, input: &[u8]) -> Option<Vec<usize>> {
        exec_many(input, &[self]).into_iter().next().unwrap()
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
    fn optional_ab() {
        // note, not anchored at start:
        let p = compile("(ab)?(ab)?(ab)?$").unwrap();
        let input = b"abababab";
        assert_eq!(p.exec(input), Some(vec![2, 8, 2, 4, 4, 6, 6, 8]));
    }

    #[test]
    fn unanchored_prefix() {
        let p = compile("(a*)b?c?$").unwrap();
        let input = b"abacba";
        assert_eq!(p.exec(input), Some(vec![5, 6, 5, 6]));
    }

    #[test]
    fn a_plus_aba() {
        let p = compile("^(?:(a+)(aba?))*$").unwrap();
        let input = b"aabaaaba";
        assert_eq!(p.exec(input), Some(vec![0, 8, 4, 5, 5, 8]));
    }

    #[test]
    fn star_star() {
        let p = compile("a**").unwrap();
        assert_eq!(p.exec(b""), Some(vec![0, 0]));
        assert_eq!(p.exec(b"a"), Some(vec![0, 1]));
        assert_eq!(p.exec(b"aa"), Some(vec![0, 2]));
    }

    #[test]
    fn nested_captures() {
        let p = compile("^((a)b)$").unwrap();
        let input = b"ab";
        assert_eq!(p.exec(input), Some(vec![0, 2, 0, 2, 0, 1]));
    }

    #[test]
    fn leftmost_greedy() {
        let p = compile("^(?:(a*)(a*)(a*))*(b+)$").unwrap();
        let input = b"aabb";
        assert_eq!(p.exec(input), Some(vec![0, 4, 0, 2, 2, 2, 2, 2, 2, 4]));
    }

    #[test]
    fn many_empty() {
        let p = compile("^((a*)(a*)(a*)|(a*)(a*)(a*)|(a*)(a*)(a*)|(a*)(a*)(a*))*$").unwrap();
        let input = b"aaa";
        assert_eq!(
            p.exec(input),
            Some(vec![
                0,
                3,
                0,
                3,
                0,
                3,
                3,
                3,
                3,
                3,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
            ])
        );
    }

    #[test]
    fn overlap_combined() {
        // report matches for both ^(abc)$ and ^a(b*)c$ on the same string
        let p0 = compile("^(abc)$").unwrap();
        let p1 = compile("^a(b*)c$").unwrap();

        assert_eq!(
            exec_many(b"abc", &[&p0, &p1]),
            vec![Some(vec![0, 3, 0, 3]), Some(vec![0, 3, 1, 2])]
        );

        assert_eq!(
            exec_many(b"ac", &[&p0, &p1]),
            vec![None, Some(vec![0, 2, 1, 1])]
        );

        assert_eq!(
            exec_many(b"abbc", &[&p0, &p1]),
            vec![None, Some(vec![0, 4, 1, 3])]
        );
    }

    #[test]
    fn pathological() {
        let p = compile("^a*a*a*a*a*a*a*a*a*b$").unwrap();
        let input = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabc";
        assert_eq!(p.exec(input), None);
    }
}
