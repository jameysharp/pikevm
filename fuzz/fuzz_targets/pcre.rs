#![no_main]
use libfuzzer_sys::fuzz_target;

use pcre2::bytes::Regex;
use pikevm::compile;

fuzz_target!(|data: (ast::Ast, ast::ASCIIString)| {
    let _ = env_logger::try_init();

    let (pattern, ast::ASCIIString(input)) = data;
    let pattern = pattern.to_string();

    if pattern.chars().filter(|&c| c == '.').count() > 5 {
        // DFA size can be exponential in the number of '.' wildcards, since by default they
        // represent a choice of two ranges ([^\n]). Limit blow-up with a rough heuristic.
        return;
    }

    // Require both PCRE and regex-syntax to accept the pattern.
    let program = compile(&pattern).unwrap();
    let regex = Regex::new(&pattern).unwrap();

    let matches = program.exec(input.as_bytes());
    let dfa = program.to_dfa();
    let dfa_matches = dfa.exec(input.as_bytes());

    let captures = regex.captures(input.as_bytes()).unwrap();
    let expected = captures.map(|captures| {
        (0..captures.len())
            .flat_map(|i| match captures.get(i) {
                Some(group) => [group.start(), group.end()],
                None => [usize::MAX, usize::MAX],
            })
            .collect()
    });

    assert_eq!(
        matches, expected,
        "wrong captures for /{pattern}/ on '{input}'",
    );

    assert_eq!(
        dfa_matches,
        expected,
        "wrong captures for DFA of /{pattern}/ on '{input}':\n{}",
        dfa.to_dot(),
    );
});

/// regex-syntax and PCRE disagree about the meaning of various odd bits of syntax. So this
/// structure exists to generate only regular expressions that both implementations ought to parse
/// the same way.
mod ast {
    use libfuzzer_sys::arbitrary::{self, Arbitrary};
    use std::collections::BTreeSet;

    pub type Ast = Alternation;

    #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
    struct Literal(u8);

    impl Arbitrary<'_> for Literal {
        fn arbitrary(u: &mut arbitrary::Unstructured) -> arbitrary::Result<Self> {
            let v: u8 = u.arbitrary()?;
            Ok(Literal(v & 0x7F))
        }

        fn size_hint(depth: usize) -> (usize, Option<usize>) {
            <u8 as Arbitrary>::size_hint(depth)
        }
    }

    #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
    struct Range(Literal, Literal);

    impl Arbitrary<'_> for Range {
        fn arbitrary(u: &mut arbitrary::Unstructured) -> arbitrary::Result<Self> {
            let (mut lo, mut hi) = u.arbitrary()?;
            if lo > hi {
                std::mem::swap(&mut lo, &mut hi);
            }
            Ok(Range(lo, hi))
        }

        fn size_hint(depth: usize) -> (usize, Option<usize>) {
            let lit_hint = <Literal as Arbitrary>::size_hint(depth);
            arbitrary::size_hint::and(lit_hint, lit_hint)
        }
    }

    #[derive(Arbitrary)]
    enum Repeatable {
        Literal(Literal),
        // Use an ordered set for character classes so a class can't both start and end with '=',
        // which PCRE treats as a POSIX "collating element". Also because it's not that interesting
        // to test different permutations of the same character class.
        Class {
            ranges: BTreeSet<Range>,
            negated: bool,
        },
        Group {
            sub: Box<Alternation>,
            capturing: bool,
        },
    }

    #[derive(Arbitrary)]
    enum RepetitionKind {
        ZeroOrOne,
        ZeroOrMore,
        OneOrMore,
    }

    #[derive(Arbitrary)]
    enum Nullable {
        Single(Repeatable),
        Repetition {
            sub: Box<Repeatable>,
            kind: RepetitionKind,
            greedy: bool,
        },
        StartLine,
        EndLine,
        StartText,
        EndText,
        WordBoundary,
        NotWordBoundary,
    }

    #[derive(Arbitrary)]
    struct Concat(Box<[Nullable]>);

    #[derive(Arbitrary)]
    pub struct Alternation(Box<[Concat]>);

    impl std::fmt::Display for Literal {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self.0 {
                b'\\' | b'.' | b'+' | b'*' | b'?' | b'(' | b')' | b'|' | b'[' | b']' | b'{'
                | b'}' | b'^' | b'$' | b'#' | b'&' | b'-' | b'~' => {
                    write!(f, "\\{}", char::from(self.0))
                }
                b'\t' => f.write_str("\\t"),
                b'\n' => f.write_str("\\n"),
                b'\r' => f.write_str("\\r"),
                32..=126 => write!(f, "{}", char::from(self.0)),
                _ => write!(f, "\\x{:02X}", self.0),
            }
        }
    }

    impl std::fmt::Display for Repeatable {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            use Repeatable::*;
            match self {
                Literal(c) => write!(f, "{}", c),
                Class { ranges, negated } => {
                    if ranges.is_empty() {
                        // Rather than convincing Arbitrary to generate only non-empty classes, I'm
                        // choosing to interpret empty classes as "anything but newline", and their
                        // negation as "anything at all".
                        if *negated {
                            f.write_str("(?s:.)")
                        } else {
                            f.write_str(".")
                        }
                    } else {
                        f.write_str("[")?;
                        if *negated {
                            f.write_str("^")?;
                        }
                        for &Range(start, end) in ranges.iter() {
                            if start == end {
                                write!(f, "{}", start)?;
                            } else {
                                write!(f, "{}-{}", start, end)?;
                            }
                        }
                        f.write_str("]")
                    }
                }
                Group { sub, capturing } => {
                    let capturing = if *capturing { "(" } else { "(?:" };
                    write!(f, "{}{})", capturing, sub)
                }
            }
        }
    }

    impl std::fmt::Display for Nullable {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            use Nullable::*;
            match self {
                Single(c) => write!(f, "{}", c),
                Repetition { sub, kind, greedy } => {
                    let kind = match kind {
                        RepetitionKind::ZeroOrOne => "?",
                        RepetitionKind::ZeroOrMore => "*",
                        RepetitionKind::OneOrMore => "+",
                    };
                    let greedy = if *greedy { "" } else { "?" };
                    write!(f, "{}{}{}", sub, kind, greedy)
                }
                StartLine => f.write_str("(?m:^)"),
                EndLine => f.write_str("(?m:$)"),
                StartText => f.write_str("\\A"),
                EndText => f.write_str("\\z"),
                WordBoundary => f.write_str("\\b"),
                NotWordBoundary => f.write_str("\\B"),
            }
        }
    }

    impl std::fmt::Display for Concat {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            self.0.iter().try_for_each(|sub| write!(f, "{}", sub))
        }
    }

    impl std::fmt::Display for Alternation {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            if !self.0.is_empty() {
                write!(f, "{}", self.0[0])?;
                for sub in &self.0[1..] {
                    write!(f, "|{}", sub)?;
                }
            }
            Ok(())
        }
    }

    impl std::fmt::Debug for Alternation {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "/{}/", self)
        }
    }

    #[derive(Clone, Debug)]
    pub struct ASCIIString(pub String);

    impl Arbitrary<'_> for ASCIIString {
        fn arbitrary(u: &mut arbitrary::Unstructured) -> arbitrary::Result<Self> {
            let input = <&[u8] as Arbitrary>::arbitrary(u)?;
            Ok(ASCIIString(
                input.iter().map(|b| char::from(b & 0x7F)).collect(),
            ))
        }
    }
}
