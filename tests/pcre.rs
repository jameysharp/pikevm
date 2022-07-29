use pcre2::bytes::Regex;
use pikevm::compile;

#[test]
fn aaaaa() {
    let units = &[
        "a", "(a)", "(a*)", "(a+)", "(a?)", "(a*?)", "(a+?)", "(a??)",
    ];
    let longest_input = "aaaaa";

    let cases = std::iter::once(String::new());
    let cases = extend(cases, &["", "^"]);
    let cases = extend(cases, units);
    let cases = extend(cases, units);
    let cases = extend(cases, units);
    let cases = extend(cases, &["", "$"]);

    for pattern in cases {
        for len in 0..longest_input.len() {
            check_pcre(&pattern, &longest_input[..len]);
        }
    }
}

fn extend<'a>(
    prefixes: impl Iterator<Item = String> + 'a,
    units: &'a [&str],
) -> impl Iterator<Item = String> + 'a {
    prefixes.flat_map(|prefix| {
        units.iter().map(move |unit| {
            let mut combined = prefix.clone();
            combined.push_str(unit);
            combined
        })
    })
}

#[test]
fn optional_ab() {
    // note, not anchored at start:
    check_pcre("(ab)?(ab)?(ab)?$", "abababab");
}

#[test]
fn unanchored_prefix() {
    check_pcre("(a*)b?c?$", "abacba");
}

#[test]
fn a_plus_aba() {
    check_pcre("^(?:(a+)(aba?))*$", "aabaaaba");
}

#[test]
fn star_star() {
    check_pcre("(?:a*)*", "");
    check_pcre("(?:a*)*", "a");
    check_pcre("(?:a*)*", "aa");
}

#[test]
fn nested_captures() {
    check_pcre("^((a)b)$", "ab");
}

#[test]
fn leftmost_greedy() {
    check_pcre("^(?:(a*)(a*)(a*))*(b+)$", "aabb");
}

#[test]
fn many_empty() {
    check_pcre("^((a*)(a*)(a*)|(a*)(a*)(a*)|(a*)(a*)(a*)|(a*)(a*)(a*))*$", "aaa");
}

fn check_pcre(pattern: &str, input: &str) {
    let program = compile(pattern).unwrap();
    let matches = program.exec(input.as_bytes()).unwrap_or_else(Vec::new);

    let expected = match Regex::new(pattern)
        .unwrap()
        .captures(input.as_bytes())
        .unwrap()
    {
        None => Vec::new(),
        Some(captures) => (0..captures.len())
            .flat_map(|i| match captures.get(i) {
                Some(group) => [group.start(), group.end()],
                None => [usize::MAX, usize::MAX],
            })
            .collect(),
    };

    assert_eq!(
        matches, expected,
        "wrong captures for /{pattern}/ on '{input}'",
    );
}
