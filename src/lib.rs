use bitvec::vec::BitVec;
use flagset::{flags, FlagSet};
use regex_syntax::hir::{self, Hir, HirKind};
use regex_syntax::utf8::Utf8Sequences;
use regex_syntax::{is_word_byte, Parser};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::rc::Rc;

pub fn compile(pat: &str) -> regex_syntax::Result<Program> {
    compile_set([pat])
}

pub fn compile_set<S: AsRef<str>>(
    pats: impl IntoIterator<Item = S>,
) -> regex_syntax::Result<Program> {
    let mut result = Program {
        buf: Vec::new(),
        registers: 2,
    };

    let hirs = pats
        .into_iter()
        .map(|pat| Parser::new().parse(pat.as_ref()))
        .collect::<regex_syntax::Result<Vec<Hir>>>()?;

    result.alts(hirs.into_iter().enumerate().map(|(idx, hir)| {
        move |p: &mut Program| {
            // Every regex in the set gets its own collection of threads during matching, and
            // every thread has its own distinct registers, so different regexes can use the
            // same register indexes without overwriting each other's capture groups.
            let registers = p.registers;
            p.registers = 0;

            if !hir.is_anchored_start() {
                p.compile_hir(&Hir::repetition(hir::Repetition {
                    kind: hir::RepetitionKind::ZeroOrMore,
                    greedy: false,
                    hir: Box::new(Hir::any(true)),
                }));
            }

            p.push(Inst::Save(0));
            p.compile_hir(&hir);
            p.push(Inst::Save(1));
            p.push(Inst::Match(idx as _));

            p.registers = registers.max(p.registers);
        }
    }));

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
    Match(u16),
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

    pub fn exec(&self, input: &[u8]) -> HashMap<u16, Vec<usize>> {
        let mut threads = Threads::new(self);
        let mut current = Vec::new();
        let mut max_threads = threads.list.len();
        let mut best = HashMap::new();

        let mut save_match = |match_idx, captures: Rc<Vec<usize>>| match best.entry(match_idx) {
            Entry::Vacant(e) => {
                e.insert(captures);
            }
            Entry::Occupied(e) => {
                let previous = e.into_mut();
                if previous[0] == captures[0] && previous[1] < captures[1] {
                    *previous = captures;
                }
            }
        };

        let mut last_sp = None;
        for (idx, sp) in input.iter().copied().enumerate() {
            threads.take(&mut current);
            if current.is_empty() {
                break;
            }

            dbg!(idx, sp, &current);
            max_threads = max_threads.max(current.len());

            for mut thread in current.drain(..) {
                dbg!(thread.pc, &self.buf[thread.pc as usize]);

                if !thread.check_assertions(last_sp, Some(sp)) {
                    continue;
                }

                match self.buf[thread.pc as usize] {
                    Inst::Range(lo, hi) => {
                        if sp >= lo && sp <= hi {
                            threads.add(idx + 1, thread.next());
                        }
                    }
                    Inst::Match(match_idx) => save_match(match_idx, thread.saved),
                    Inst::Assertion(_) | Inst::Save(_) | Inst::Jmp(_) | Inst::Split { .. } => {
                        unreachable!()
                    }
                }
            }

            last_sp = Some(sp);
        }

        dbg!(max_threads, self.registers);

        debug_assert!(current.is_empty());
        drop(current);
        let Threads { list: current, .. } = threads;

        for mut thread in current {
            if let Inst::Match(match_idx) = self.buf[thread.pc as usize] {
                if thread.check_assertions(last_sp, None) {
                    save_match(match_idx, thread.saved);
                }
            }
        }
        best.into_iter()
            .map(|(match_idx, saved)| (match_idx, (*saved).clone()))
            .collect()
    }
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
            Inst::Range(_, _) | Inst::Match(_) => {
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

        let matches = p.exec(input);
        assert_eq!(matches, HashMap::from([(0, vec![2, 8, 2, 4, 4, 6, 6, 8])]));
    }

    #[test]
    fn unanchored_prefix() {
        let p = compile("(a*)b?c?$").unwrap();
        let input = b"abacba";

        let matches = p.exec(input);
        assert_eq!(matches, HashMap::from([(0, vec![5, 6, 5, 6])]));
    }

    #[test]
    fn a_plus_aba() {
        let p = compile("^(?:(a+)(aba?))*$").unwrap();
        let input = b"aabaaaba";

        let matches = p.exec(input);
        assert_eq!(matches, HashMap::from([(0, vec![0, 8, 4, 5, 5, 8])]));
    }

    #[test]
    fn star_star() {
        let p = compile("a**").unwrap();
        assert_eq!(p.exec(b""), HashMap::from([(0, vec![0, 0])]));
        assert_eq!(p.exec(b"a"), HashMap::from([(0, vec![0, 1])]));
        assert_eq!(p.exec(b"aa"), HashMap::from([(0, vec![0, 2])]));
    }

    #[test]
    fn nested_captures() {
        let p = compile("^((a)b)$").unwrap();
        let input = b"ab";

        let matches = p.exec(input);
        assert_eq!(matches, HashMap::from([(0, vec![0, 2, 0, 2, 0, 1])]));
    }

    #[test]
    fn leftmost_greedy() {
        let p = compile("^(?:(a*)(a*)(a*))*(b+)$").unwrap();
        let input = b"aabb";

        let matches = p.exec(input);
        assert_eq!(
            matches,
            HashMap::from([(0, vec![0, 4, 0, 2, 2, 2, 2, 2, 2, 4])])
        );
    }

    #[test]
    fn many_empty() {
        let p = compile("^((a*)(a*)(a*)|(a*)(a*)(a*)|(a*)(a*)(a*)|(a*)(a*)(a*))*$").unwrap();
        let input = b"aaa";

        let matches = p.exec(input);
        assert_eq!(
            matches,
            HashMap::from([(
                0,
                vec![
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
                ]
            )])
        );
    }

    #[test]
    fn overlap_combined() {
        // report matches for both ^(abc)$ and ^a(b*)c$ on the same string
        let p = compile_set(&["^(abc)$", "^a(b*)c$"]).unwrap();

        let matches = p.exec(b"abc");
        assert_eq!(
            matches,
            HashMap::from([(0, vec![0, 3, 0, 3]), (1, vec![0, 3, 1, 2]),])
        );

        let matches = p.exec(b"ac");
        assert_eq!(matches, HashMap::from([(1, vec![0, 2, 1, 1]),]));

        let matches = p.exec(b"abbc");
        assert_eq!(matches, HashMap::from([(1, vec![0, 4, 1, 3]),]));
    }

    #[test]
    fn pathological() {
        let p = compile("^a*a*a*a*a*a*a*a*a*b$").unwrap();
        let input = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabc";

        assert_eq!(p.exec(input), HashMap::new());
    }
}
