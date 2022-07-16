use std::collections::{HashMap, HashSet};
use std::rc::Rc;

#[derive(Clone, Debug)]
pub enum Regex {
    Range(u8, u8),
    Capture(Box<Regex>),
    Seq(Vec<Regex>),
    Alt(Box<(Regex, Regex)>),
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
        let (last, init) = pats.split_last().unwrap();

        for (idx, pat) in init.iter().enumerate() {
            let split = result.placeholder();
            let l1 = result.loc();
            result.compile(pat);
            result.push(Inst::Match(idx as _));
            let l2 = result.loc();
            result.patch(split, Inst::Split(l1, l2));
        }

        result.compile(last);
        result.push(Inst::Match(init.len() as _));

        dbg!(&result);
        result
    }
}

impl std::ops::BitOr for Regex {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Regex::Alt(Box::new((self, rhs)))
    }
}

#[derive(Clone, Debug)]
enum Inst {
    Range(u8, u8),
    Match(u16),
    Save(u16),
    Jmp(u16),
    Split(u16, u16),
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
                let split = self.placeholder();
                let l1 = self.loc();
                self.compile(&pats.0);
                let jmp = self.placeholder();
                let l2 = self.loc();
                self.compile(&pats.1);
                let l3 = self.loc();
                self.patch(split, Inst::Split(l1, l2));
                self.patch(jmp, Inst::Jmp(l3));
            }
            Regex::Opt(pat, greedy) => {
                let split = self.placeholder();
                let l1 = self.loc();
                self.compile(pat);
                let l2 = self.loc();
                if *greedy {
                    self.patch(split, Inst::Split(l1, l2));
                } else {
                    self.patch(split, Inst::Split(l2, l1));
                }
            }
            Regex::Star(pat, greedy) => {
                let split = self.placeholder();
                let l2 = self.loc();
                self.compile(pat);
                self.push(Inst::Jmp(split));
                let l3 = self.loc();
                if *greedy {
                    self.patch(split, Inst::Split(l2, l3));
                } else {
                    self.patch(split, Inst::Split(l3, l2));
                }
            }
            Regex::Plus(pat, greedy) => {
                let l1 = self.loc();
                self.compile(pat);
                let l3 = self.loc() + 1;
                if *greedy {
                    self.push(Inst::Split(l1, l3));
                } else {
                    self.push(Inst::Split(l3, l1));
                }
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
                    Inst::Save(_) | Inst::Jmp(_) | Inst::Split(_, _) => unreachable!(),
                }
            }
        }

        max_threads = max_threads.max(threads.list.len());
        dbg!(max_threads, self.registers);

        let mut best = HashMap::new();
        threads.take(&mut current);
        for thread in current.drain(..) {
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
    active: HashSet<u16>,
    list: Vec<Thread>,
}

impl<'a> Threads<'a> {
    fn new(program: &Program) -> Threads {
        let mut saved = Vec::new();
        saved.resize(program.registers as usize, usize::MAX);
        let mut result = Threads {
            program,
            active: HashSet::new(),
            list: Vec::new(),
        };
        result.add(0, 0, Rc::new(saved));
        result
    }

    fn take(&mut self, buf: &mut Vec<Thread>) {
        std::mem::swap(&mut self.list, buf);
        self.active.clear();
    }

    fn add(&mut self, idx: usize, pc: u16, mut saved: Rc<Vec<usize>>) {
        if !self.active.insert(pc) {
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
            Inst::Split(a, b) => {
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
        let ab = Regex::seq([Regex::byte(b'a'), Regex::byte(b'b')]).capture().opt();
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
            ]).capture(),
        ]).star();
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
        assert_eq!(matches, HashMap::from([(0, vec![
            0, 3,
            0, 3,
            3, 3,
            3, 3,
            usize::MAX, usize::MAX,
            usize::MAX, usize::MAX,
            usize::MAX, usize::MAX,
            usize::MAX, usize::MAX,
            usize::MAX, usize::MAX,
            usize::MAX, usize::MAX,
            usize::MAX, usize::MAX,
            usize::MAX, usize::MAX,
            usize::MAX, usize::MAX,
        ])]));
    }

    #[test]
    fn overlap_combined() {
        // report matches for both ^(abc)$ and ^(ab*c)$ on the same string
        let r0 = Regex::seq([Regex::byte(b'a'), Regex::byte(b'b'), Regex::byte(b'c')]).capture();
        let r1 = Regex::seq([Regex::byte(b'a'), Regex::byte(b'b').star(), Regex::byte(b'c')]).capture();
        let p = Regex::compile_set(&[r0, r1]);

        let matches = p.fullmatch(b"abc");
        assert_eq!(matches, HashMap::from([
            (0, vec![0, 3, usize::MAX, usize::MAX]),
            (1, vec![usize::MAX, usize::MAX, 0, 3]),
        ]));

        let matches = p.fullmatch(b"ac");
        assert_eq!(matches, HashMap::from([
            (1, vec![usize::MAX, usize::MAX, 0, 2]),
        ]));

        let matches = p.fullmatch(b"abbc");
        assert_eq!(matches, HashMap::from([
            (1, vec![usize::MAX, usize::MAX, 0, 4]),
        ]));
    }

    #[test]
    fn pathological() {
        // ^a*a*a*a*a*a*a*a*a*b$
        let a = Regex::star(Regex::byte(b'a'));
        let r = Regex::seq([a.clone(), a.clone(), a.clone(), a.clone(), a.clone(), a.clone(), a.clone(), a.clone(), a, Regex::byte(b'b')]);
        let input = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabc";

        assert_eq!(r.compile().fullmatch(input), HashMap::new());
    }
}
