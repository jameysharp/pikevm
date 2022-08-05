use bitvec::vec::BitVec;
use log::{debug, trace};
use regex_syntax::hir::{self, Hir};
use regex_syntax::utf8::Utf8Sequences;
use regex_syntax::{is_word_byte, ParserBuilder};
use std::borrow::Borrow;
use std::rc::Rc;

pub mod dfa;

#[derive(Clone, Debug)]
pub enum Error {
    Syntax(regex_syntax::Error),
    Compile(CompileError),
}

impl From<regex_syntax::Error> for Error {
    fn from(e: regex_syntax::Error) -> Error {
        Error::Syntax(e)
    }
}

impl From<CompileError> for Error {
    fn from(e: CompileError) -> Error {
        Error::Compile(e)
    }
}

pub fn compile(pat: &str) -> Result<Program, Error> {
    let mut parse_config = ParserBuilder::new();
    parse_config.allow_invalid_utf8(true);
    parse_config.unicode(false);

    let mut hir = parse_config.build().parse(pat)?;

    // Wrap capture group 0 around the whole pattern.
    hir = Hir::group(hir::Group {
        kind: hir::GroupKind::CaptureIndex(0),
        hir: Box::new(hir),
    });

    // If the pattern is not (always) anchored at the start, insert a non-greedy match to skip
    // anything at the beginning that doesn't match the pattern.
    if !hir.is_anchored_start() {
        hir = Hir::concat(vec![
            Hir::repetition(hir::Repetition {
                kind: hir::RepetitionKind::ZeroOrMore,
                greedy: false,
                hir: Box::new(Hir::any(true)),
            }),
            hir,
        ]);
    }

    let result = hir::visit(&hir, Compiler::default())?;
    debug!("Pike VM program for /{}/: {:#?}", pat, &result);
    Ok(result)
}

#[derive(Clone, Copy, Debug)]
enum Assertions {
    StartLine,
    EndLine,
    StartText,
    EndText,
    AsciiWordBoundary,
    AsciiNotWordBoundary,
}

#[derive(Clone)]
enum Inst {
    Range(u8, u8),
    Assertion(Assertions),
    Match,
    Save(u16),
    Jmp(u16),
    Split { prefer_next: bool, to: u16 },
}

impl std::fmt::Debug for Inst {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Inst::Range(lo, hi) => write!(f, "{:?}-{:?}", char::from(lo), char::from(hi)),
            Inst::Assertion(kind) => write!(f, "assert {:?}", kind),
            Inst::Match => write!(f, "match"),
            Inst::Save(reg) => write!(f, "save -> register {}", reg),
            Inst::Jmp(to) => write!(f, "jmp {}", to),
            Inst::Split { prefer_next, to } => {
                if prefer_next {
                    write!(f, "split next then {}", to)
                } else {
                    write!(f, "split {} then next", to)
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct Program {
    buf: Box<[Inst]>,
    registers: u16,
}

impl std::fmt::Debug for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Program ({} registers) ", self.registers)?;
        f.debug_map().entries(self.buf.iter().enumerate()).finish()
    }
}

impl Program {
    pub fn exec(&self, input: &[u8]) -> Option<Vec<usize>> {
        exec_many(input, &[self]).into_iter().next().unwrap()
    }

    pub fn to_dfa(&self) -> dfa::DFA {
        dfa::DFA::new(self)
    }
}

#[derive(Clone, Debug, Default)]
struct Compiler {
    buf: Vec<Inst>,
    registers: u16,
    in_nullable_rep: BitVec,
    captures: Vec<Vec<u16>>,
    labels: Vec<u16>,
    alternations: Vec<AlternationState>,
}

impl Compiler {
    fn loc(&self) -> u16 {
        self.buf.len() as u16
    }

    fn push(&mut self, inst: Inst) {
        self.buf.push(inst);
    }

    fn placeholder(&mut self) -> u16 {
        let loc = self.loc();
        self.push(Inst::Jmp(u16::MAX));
        loc
    }

    fn patch(&mut self, loc: u16, inst: Inst) {
        self.buf[loc as usize] = inst;
    }

    /// Are we in a repetition of a subexpression which can match the empty string? If so, PCRE
    /// reports any capture groups in this subexpression as matching after the end of the
    /// repetition, as if the repetition ran one extra time at the end.
    fn in_nullable_rep(&self) -> bool {
        *self.in_nullable_rep.last().as_deref().unwrap_or(&false)
    }

    fn repetition(&mut self, rep: &hir::Repetition, end: bool) -> Result<(), CompileError> {
        match rep.kind {
            hir::RepetitionKind::ZeroOrOne => {
                if !end {
                    let split = self.placeholder();
                    self.labels.push(split);
                } else {
                    let split = self.labels.pop().unwrap();
                    let to = self.loc();
                    let prefer_next = rep.greedy;
                    self.patch(split, Inst::Split { prefer_next, to });
                }
            }
            hir::RepetitionKind::ZeroOrMore => {
                if !end {
                    let split = self.placeholder();
                    self.labels.push(split);
                } else {
                    let split = self.labels.pop().unwrap();
                    self.push(Inst::Jmp(split));
                    let to = self.loc();
                    let prefer_next = rep.greedy;
                    self.patch(split, Inst::Split { prefer_next, to });
                }
            }
            hir::RepetitionKind::OneOrMore => {
                if let hir::HirKind::Repetition(_) = rep.hir.kind() {
                    // PCRE steals syntax like '++' for "possessive" quantifiers. I don't want to
                    // implement those, so just refuse to compile them.
                    return Err(CompileError);
                }
                if !end {
                    let to = self.loc();
                    self.labels.push(to);
                } else {
                    let to = self.labels.pop().unwrap();
                    let prefer_next = !rep.greedy;
                    self.push(Inst::Split { prefer_next, to });
                }
            }
            hir::RepetitionKind::Range(_) => return Err(CompileError),
        }

        if !end {
            if rep.hir.is_match_empty() && !self.in_nullable_rep() {
                self.captures.push(Vec::new());
            }
            self.in_nullable_rep.push(rep.hir.is_match_empty());
        } else {
            self.in_nullable_rep.pop().unwrap();
            if rep.hir.is_match_empty() && !self.in_nullable_rep() {
                let deferred = self.captures.pop().unwrap();
                self.buf.extend(deferred.into_iter().map(Inst::Save));
            }
        }
        Ok(())
    }

    fn save(&mut self, group: &hir::GroupKind, end: bool) {
        match group {
            hir::GroupKind::CaptureIndex(index) | hir::GroupKind::CaptureName { index, .. } => {
                let mut register = (index * 2).try_into().unwrap();
                if !end {
                    self.registers = register + 2;
                } else {
                    register += 1;
                }
                if self.in_nullable_rep() {
                    self.captures.last_mut().unwrap().push(register);
                } else {
                    self.push(Inst::Save(register));
                }
            }
            hir::GroupKind::NonCapturing => {}
        }
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
}

#[derive(Clone, Debug)]
pub struct CompileError;

impl hir::Visitor for Compiler {
    type Output = Program;
    type Err = CompileError;

    fn visit_pre(&mut self, hir: &Hir) -> Result<(), CompileError> {
        use hir::HirKind::*;
        match hir.kind() {
            Empty => {}
            Literal(hir::Literal::Unicode(u)) => self.buf.extend(
                u.encode_utf8(&mut [0; 4])
                    .bytes()
                    .map(|b| Inst::Range(b, b)),
            ),
            &Literal(hir::Literal::Byte(b)) => self.push(Inst::Range(b, b)),
            Class(hir::Class::Unicode(uc)) => {
                let seqs = uc
                    .iter()
                    .flat_map(|range| Utf8Sequences::new(range.start(), range.end()));
                self.alts(seqs, |p, seq| {
                    for byte_range in seq.as_slice() {
                        p.push(Inst::Range(byte_range.start, byte_range.end));
                    }
                });
            }
            Class(hir::Class::Bytes(bc)) => self.alts(bc.iter(), |p, range| {
                p.push(Inst::Range(range.start(), range.end()));
            }),
            Anchor(kind) => self.push(Inst::Assertion(match kind {
                hir::Anchor::StartLine => Assertions::StartLine,
                hir::Anchor::EndLine => Assertions::EndLine,
                hir::Anchor::StartText => Assertions::StartText,
                hir::Anchor::EndText => Assertions::EndText,
            })),
            WordBoundary(kind) => self.push(Inst::Assertion(match kind {
                hir::WordBoundary::Ascii => Assertions::AsciiWordBoundary,
                hir::WordBoundary::AsciiNegate => Assertions::AsciiNotWordBoundary,
                // FIXME: Unicode word boundaries are hard when matching byte-at-a-time. This
                // implementation cheats by pretending you asked for ASCII boundaries instead.
                hir::WordBoundary::Unicode => Assertions::AsciiWordBoundary,
                hir::WordBoundary::UnicodeNegate => Assertions::AsciiNotWordBoundary,
            })),
            Repetition(rep) => self.repetition(rep, false)?,
            Group(group) => self.save(&group.kind, false),
            Concat(_) => {}
            Alternation(subs) => {
                let mut state = AlternationState::begin(self, subs);
                state.begin_alternative(self);
                self.alternations.push(state);
            }
        }
        Ok(())
    }

    fn visit_post(&mut self, hir: &Hir) -> Result<(), CompileError> {
        use hir::HirKind::*;
        match hir.kind() {
            // Non-recursive cases are handled above in visit_pre.
            Empty | Literal(_) | Class(_) | Anchor(_) | WordBoundary(_) => {}
            Repetition(rep) => self.repetition(rep, true)?,
            Group(group) => self.save(&group.kind, true),
            Concat(_) => {}
            Alternation(_) => {
                let mut state = self.alternations.pop().unwrap();
                state.end_alternative(self);
                state.end(self);
            }
        }
        Ok(())
    }

    fn visit_alternation_in(&mut self) -> Result<(), CompileError> {
        let mut state = self.alternations.pop().unwrap();
        state.end_alternative(self);
        state.begin_alternative(self);
        self.alternations.push(state);
        Ok(())
    }

    fn finish(mut self) -> Result<Program, CompileError> {
        debug_assert!(self.in_nullable_rep.is_empty());
        debug_assert!(self.captures.is_empty());
        debug_assert!(self.labels.is_empty());
        debug_assert!(self.alternations.is_empty());
        self.push(Inst::Match);
        Ok(Program {
            buf: self.buf.into_boxed_slice(),
            registers: self.registers,
        })
    }
}

#[derive(Clone, Debug)]
struct AlternationState {
    remaining: usize,
    first_nullable: usize,
    jmps: Vec<u16>,
}

impl AlternationState {
    fn begin(compiler: &Compiler, subs: &Vec<Hir>) -> AlternationState {
        let mut state = AlternationState {
            remaining: subs.len(),
            first_nullable: 0,
            jmps: Vec::with_capacity(subs.len() - 1),
        };
        if compiler.in_nullable_rep() {
            // We're inside a nullable subexpression, so some alternative is nullable.
            let i = subs.iter().position(|sub| sub.is_match_empty()).unwrap();
            state.first_nullable = i + 1;
        }
        state
    }

    fn begin_alternative(&mut self, compiler: &mut Compiler) {
        compiler.in_nullable_rep.push(self.first_nullable == 1);
        if self.first_nullable > 0 {
            self.first_nullable -= 1;
        }

        if self.remaining > 1 {
            let split = compiler.placeholder();
            compiler.labels.push(split);
        }
    }

    fn end_alternative(&mut self, compiler: &mut Compiler) {
        if self.remaining > 1 {
            let split = compiler.labels.pop().unwrap();
            self.jmps.push(compiler.placeholder());
            let to = compiler.loc();
            let prefer_next = true; // prefer left-most successful alternative
            compiler.patch(split, Inst::Split { prefer_next, to });
        }
        self.remaining -= 1;

        compiler.in_nullable_rep.pop().unwrap();
    }

    fn end(self, compiler: &mut Compiler) {
        debug_assert_eq!(self.remaining, 0);
        debug_assert_eq!(self.first_nullable, 0);
        let to = compiler.loc();
        for jmp in self.jmps {
            compiler.patch(jmp, Inst::Jmp(to));
        }
    }
}

pub fn exec_many(input: &[u8], patterns: &[impl Borrow<Program>]) -> Vec<Option<Vec<usize>>> {
    let mut patterns = patterns
        .iter()
        .map(|pattern| Threads::new(pattern.borrow()))
        .collect::<Vec<_>>();

    let mut last_sp = None;
    for (idx, sp) in input.iter().copied().enumerate() {
        trace!("input[{}] = {:?}", idx, char::from(sp));

        let mut progress = false;
        for threads in patterns.iter_mut() {
            progress |= threads.step(idx, last_sp, Some(sp));
        }

        // If every pattern is stuck (no more threads can match), we can stop processing the input.
        if !progress {
            break;
        }

        last_sp = Some(sp);
    }

    patterns
        .into_iter()
        .map(|mut threads| {
            // Note that if we stopped due to all patterns getting stuck, then `step` is a no-op.
            threads.step(input.len(), last_sp, None);
            threads.result.map(|rc| rc.to_vec())
        })
        .collect()
}

#[derive(Clone, Debug)]
struct Thread {
    pc: u16,
    saved: Rc<Vec<usize>>,
}

impl Thread {
    fn next(mut self) -> Self {
        self.pc += 1;
        self
    }
}

struct Threads<'a> {
    program: &'a Program,
    active: BitVec,
    list: Vec<Thread>,
    result: Option<Rc<Vec<usize>>>,
}

impl<'a> Threads<'a> {
    fn new(program: &Program) -> Threads {
        let mut result = Threads {
            program,
            active: BitVec::repeat(false, program.buf.len()),
            list: Vec::new(),
            result: None,
        };
        result.list.push(Thread {
            pc: 0,
            saved: Rc::new(vec![usize::MAX; program.registers.into()]),
        });
        result
    }

    fn step(&mut self, idx: usize, prev: Option<u8>, next: Option<u8>) -> bool {
        if self.list.is_empty() {
            return false;
        }
        self.active.fill(false);
        for thread in std::mem::take(&mut self.list) {
            if self.add(idx, thread, prev, next) {
                // found a Match, which supersedes any lower-priority threads
                break;
            }
        }
        true
    }

    fn add(&mut self, idx: usize, mut thread: Thread, prev: Option<u8>, next: Option<u8>) -> bool {
        if self.active.replace(thread.pc as usize, true) {
            return false;
        }
        // NFA epsilon closure: a thread can only stop on Range or Match instructions. For anything
        // else, recurse on the targets of the instruction. Note that this recursion is at worst
        // O(n) in the number of instructions; any epsilon cycles are broken using self.active.
        trace!(
            "eval {:?} ({:?})",
            &self.program.buf[thread.pc as usize],
            &thread
        );
        match self.program.buf[thread.pc as usize] {
            Inst::Range(lo, hi) => {
                if let Some(sp) = next {
                    if sp >= lo && sp <= hi {
                        self.list.push(thread.next());
                    }
                }
                false
            }
            Inst::Match => {
                self.result = Some(thread.saved);
                trace!("match: {:?}", &self.result);
                // unwind, discarding all lower-priority threads
                true
            }
            Inst::Assertion(kind) => {
                let passed = match kind {
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
                // short-circuit if the assertion failed
                passed && self.add(idx, thread.next(), prev, next)
            }
            Inst::Save(reg) => {
                Rc::make_mut(&mut thread.saved)[reg as usize] = idx;
                self.add(idx, thread.next(), prev, next)
            }
            Inst::Jmp(pc) => self.add(idx, Thread { pc, ..thread }, prev, next),
            Inst::Split { prefer_next, to } => {
                let mut a = thread.clone().next();
                let mut b = Thread { pc: to, ..thread };
                if !prefer_next {
                    std::mem::swap(&mut a, &mut b);
                }
                // short-circuit if the first thread gets a Match
                self.add(idx, a, prev, next) || self.add(idx, b, prev, next)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn pathological() {
        let _ = env_logger::try_init();

        let p = compile("^a*a*a*a*a*a*a*a*a*b$").unwrap();
        let input = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabc";
        assert_eq!(p.exec(input), None);
    }
}
