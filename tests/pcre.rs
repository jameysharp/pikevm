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
fn misordered() {
    check_pcre("$^", "");
    check_pcre("$^", "a");
}

#[test]
fn optional_failed_assertion() {
    check_pcre("^a(?:^)?b$", "ab");
    check_pcre("^a(?:\\b)?b$", "ab");
    check_pcre("^a(?:\\B)? $", "a ");
    check_pcre("(?:\\b\\B)?", "ab");
    check_pcre("\\B\\z|\\z", "a");
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
fn nullable_captures() {
    check_pcre("(a*)+", "aaa");
    check_pcre("^((a*)(a*)(a*))*", "aaa");
    check_pcre("((a)*(a)?a*)+", "aaa");
}

#[test]
fn leftmost_greedy() {
    check_pcre("^(?:(a*)(a*)(a*))*(b+)$", "b");
    check_pcre("^(?:(a*)(a*)(a*))*(b+)$", "ab");
    check_pcre("^(?:(a*)(a*)(a*))*(b+)$", "aabb");
}

#[test]
fn many_empty() {
    check_pcre(
        "^((a*)(a*)(a*)|(a*)(a*)(a*)|(a*)(a*)(a*)|(a*)(a*)(a*))*$",
        "aaa",
    );
}

fn check_pcre(pattern: &str, input: &str) {
    let _ = env_logger::try_init();

    let program = compile(pattern).unwrap();
    let matches = program.exec(input.as_bytes());
    let dfa = program.to_dfa();
    let dfa_matches = dfa.exec(input.as_bytes());

    let regex = Regex::new(pattern).unwrap();
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
}
