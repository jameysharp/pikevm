use bitvec::vec::BitVec;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub enum Regex {
    Range(u8, u8),
    Capture(Box<Regex>),
    Seq(Vec<Regex>),
    Alt(Vec<Regex>),
    Opt(Box<Regex>, bool),
    Star(Box<Regex>, bool),
    Plus(Box<Regex>, bool),
}

impl Regex {
    pub fn any() -> Regex {
        Regex::Range(0, 255)
    }

    pub fn byte(b: u8) -> Regex {
        Regex::Range(b, b)
    }

    pub fn capture(self) -> Regex {
        Regex::Capture(Box::new(self))
    }

    pub fn seq(pats: impl IntoIterator<Item = Regex>) -> Regex {
        Regex::Seq(pats.into_iter().collect())
    }

    pub fn alternation(pats: Vec<Regex>) -> Regex {
        Regex::Alt(pats)
    }

    pub fn opt(self) -> Regex {
        Regex::Opt(Box::new(self), true)
    }

    pub fn opt_nongreedy(self) -> Regex {
        Regex::Opt(Box::new(self), false)
    }

    pub fn star(self) -> Regex {
        Regex::Star(Box::new(self), true)
    }

    pub fn star_nongreedy(self) -> Regex {
        Regex::Star(Box::new(self), false)
    }

    pub fn plus(self) -> Regex {
        Regex::Plus(Box::new(self), true)
    }

    pub fn plus_nongreedy(self) -> Regex {
        Regex::Plus(Box::new(self), false)
    }

    pub fn compile(self) -> Program {
        Regex::compile_set(&[self])
    }

    pub fn compile_set(pats: &[Regex]) -> Program {
        let mut result = Program {
            buf: Vec::new(),
            registers: 0,
        };

        result.alts(pats.iter().enumerate().map(|(idx, pat)| {
            move |p: &mut Program| {
                // Every regex in the set gets its own collection of threads during matching, and
                // every thread has its own distinct registers, so different regexes can use the
                // same register indexes without overwriting each other's capture groups.
                let registers = p.registers;
                p.registers = 0;

                p.compile(pat);
                p.push(Inst::Match(idx as _));

                p.registers = registers.max(p.registers);
            }
        }));

        dbg!(&result);
        result
    }
}

impl std::ops::BitOr for Regex {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Regex::Alt(vec![self, rhs])
    }
}

#[derive(Clone, Debug)]
enum Inst {
    Range(u8, u8),
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

    fn compile(&mut self, pat: &Regex) {
        match pat {
            &Regex::Range(lo, hi) => {
                self.push(Inst::Range(lo, hi));
            }
            Regex::Capture(pat) => {
                let register = self.registers;
                self.registers += 2;
                self.push(Inst::Save(register));
                self.compile(pat);
                self.push(Inst::Save(register + 1));
            }
            Regex::Seq(pats) => {
                for pat in pats {
                    self.compile(pat);
                }
            }
            Regex::Alt(pats) => {
                self.alts(pats.iter().map(|pat| |p: &mut Program| p.compile(pat)));
            }
            &Regex::Opt(ref pat, prefer_next) => {
                let split = self.placeholder();
                self.compile(pat);
                let to = self.loc();
                self.patch(split, Inst::Split { prefer_next, to });
            }
            &Regex::Star(ref pat, prefer_next) => {
                let split = self.placeholder();
                self.compile(pat);
                self.push(Inst::Jmp(split));
                let to = self.loc();
                self.patch(split, Inst::Split { prefer_next, to });
            }
            &Regex::Plus(ref pat, greedy) => {
                let to = self.loc();
                self.compile(pat);
                self.push(Inst::Split {
                    prefer_next: !greedy,
                    to,
                });
            }
        }
    }

    pub fn fullmatch(&self, input: &[u8]) -> HashMap<u16, Vec<usize>> {
        let mut threads = Threads::new(self);
        let mut current = Vec::new();
        let mut max_threads = 0;

        for (idx, sp) in input.iter().enumerate() {
            dbg!(idx, *sp, &threads.list);
            max_threads = max_threads.max(threads.list.len());

            threads.take(&mut current);

            for thread in current.drain(..) {
                dbg!(thread.pc, &self.buf[thread.pc as usize]);
                match self.buf[thread.pc as usize] {
                    Inst::Range(lo, hi) => {
                        if *sp >= lo && *sp <= hi {
                            threads.add(idx + 1, thread.pc + 1, thread.saved);
                        }
                    }
                    Inst::Match(_) => {
                        // We're trying to match the full input string, so discard any match which
                        // completes before the end.
                    }
                    Inst::Save(_) | Inst::Jmp(_) | Inst::Split { .. } => unreachable!(),
                }
            }
        }

        max_threads = max_threads.max(threads.list.len());
        dbg!(max_threads, self.registers);

        debug_assert!(current.is_empty());
        drop(current);
        let Threads { list: current, .. } = threads;

        let mut best = HashMap::new();
        for thread in current {
            if let Inst::Match(idx) = self.buf[thread.pc as usize] {
                best.entry(idx).or_insert_with(|| (*thread.saved).clone());
            }
        }

        best
    }
}

#[derive(Clone, Debug)]
struct Thread {
    pc: u16,
    saved: Rc<Vec<usize>>,
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
        result.add(0, 0, Rc::new(saved));
        result
    }

    fn take(&mut self, buf: &mut Vec<Thread>) {
        std::mem::swap(&mut self.list, buf);
        self.active.fill(false);
    }

    fn add(&mut self, idx: usize, pc: u16, mut saved: Rc<Vec<usize>>) {
        if self.active.replace(pc as usize, true) {
            return;
        }
        // NFA epsilon closure: a thread can only stop on Range or Match instructions. For anything
        // else, recurse on the targets of the instruction. Note that this recursion is at worst
        // O(n) in the number of instructions; any epsilon cycles are broken using self.active.
        match self.program.buf[pc as usize] {
            Inst::Range(_, _) | Inst::Match(_) => {
                self.list.push(Thread { pc, saved });
            }
            Inst::Save(reg) => {
                Rc::make_mut(&mut saved)[reg as usize] = idx;
                self.add(idx, pc + 1, saved);
            }
            Inst::Jmp(to) => self.add(idx, to, saved),
            Inst::Split { prefer_next, to } => {
                let (a, b) = if prefer_next {
                    (pc + 1, to)
                } else {
                    (to, pc + 1)
                };
                self.add(idx, a, saved.clone());
                self.add(idx, b, saved);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn optional_ab() {
        // (ab)?(ab)?(ab)?$ (not anchored at start)
        let ab = Regex::seq([Regex::byte(b'a'), Regex::byte(b'b')])
            .capture()
            .opt();
        let r = Regex::seq([Regex::any().star_nongreedy(), ab.clone(), ab.clone(), ab]);
        let input = b"abababab";

        let matches = r.compile().fullmatch(input);
        assert_eq!(matches, HashMap::from([(0, vec![2, 4, 4, 6, 6, 8])]));
    }

    #[test]
    fn unanchored_prefix() {
        // (a*)b?c?$ (not anchored at start)
        let r = Regex::seq([
            Regex::any().star_nongreedy(),
            Regex::byte(b'a').star().capture(),
            Regex::byte(b'b').opt(),
            Regex::byte(b'c').opt(),
        ]);
        let input = b"abacba";

        let matches = r.compile().fullmatch(input);
        assert_eq!(matches, HashMap::from([(0, vec![5, 6])]));
    }

    #[test]
    fn a_plus_aba() {
        // ^(?:(a+)(aba?))*$
        let r = Regex::seq([
            Regex::byte(b'a').plus().capture(),
            Regex::seq([
                Regex::byte(b'a'),
                Regex::byte(b'b'),
                Regex::byte(b'a').opt(),
            ])
            .capture(),
        ])
        .star();
        let input = b"aabaaaba";

        let matches = r.compile().fullmatch(input);
        assert_eq!(matches, HashMap::from([(0, vec![4, 5, 5, 8])]));
    }

    #[test]
    fn star_star() {
        let p = Regex::byte(b'a').star().star().compile();

        assert_eq!(p.fullmatch(b""), HashMap::from([(0, vec![])]));

        assert_eq!(p.fullmatch(b"a"), HashMap::from([(0, vec![])]));

        assert_eq!(p.fullmatch(b"aa"), HashMap::from([(0, vec![])]));
    }

    #[test]
    fn nested_captures() {
        // ^((a)b)$
        let r = Regex::seq([Regex::byte(b'a').capture(), Regex::byte(b'b')]).capture();
        let input = b"ab";

        let matches = r.compile().fullmatch(input);
        assert_eq!(matches, HashMap::from([(0, vec![0, 2, 0, 1])]));
    }

    #[test]
    fn leftmost_greedy() {
        // ^(?:(a*)(a*)(a*))*(b+)$
        let a_star = Regex::byte(b'a').star().capture();
        let r = Regex::seq([
            Regex::seq([a_star.clone(), a_star.clone(), a_star]).star(),
            Regex::byte(b'b').plus().capture(),
        ]);
        let input = b"aabb";

        let matches = r.compile().fullmatch(input);
        assert_eq!(matches, HashMap::from([(0, vec![0, 2, 2, 2, 2, 2, 2, 4])]));
    }

    #[test]
    fn many_empty() {
        // ^((a*)(a*)(a*)|(a*)(a*)(a*)|(a*)(a*)(a*)|(a*)(a*)(a*))*$
        let a_star = Regex::byte(b'a').star().capture();
        let triple = Regex::seq([a_star.clone(), a_star.clone(), a_star]);
        let alts = triple.clone() | triple.clone() | triple.clone() | triple;
        let r = alts.capture().star();
        let input = b"aaa";

        let matches = r.compile().fullmatch(input);
        assert_eq!(
            matches,
            HashMap::from([(
                0,
                vec![
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
        let r0 = Regex::seq([Regex::byte(b'a'), Regex::byte(b'b'), Regex::byte(b'c')]).capture();
        let r1 = Regex::seq([
            Regex::byte(b'a'),
            Regex::byte(b'b').star().capture(),
            Regex::byte(b'c'),
        ]);
        let p = Regex::compile_set(&[r0, r1]);

        let matches = p.fullmatch(b"abc");
        assert_eq!(matches, HashMap::from([(0, vec![0, 3]), (1, vec![1, 2]),]));

        let matches = p.fullmatch(b"ac");
        assert_eq!(matches, HashMap::from([(1, vec![1, 1]),]));

        let matches = p.fullmatch(b"abbc");
        assert_eq!(matches, HashMap::from([(1, vec![1, 3]),]));
    }

    #[test]
    fn pathological() {
        // ^a*a*a*a*a*a*a*a*a*b$
        let a = Regex::star(Regex::byte(b'a'));
        let r = Regex::seq([
            a.clone(),
            a.clone(),
            a.clone(),
            a.clone(),
            a.clone(),
            a.clone(),
            a.clone(),
            a.clone(),
            a,
            Regex::byte(b'b'),
        ]);
        let input = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabc";

        assert_eq!(r.compile().fullmatch(input), HashMap::new());
    }
}
