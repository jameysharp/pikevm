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
            .flat_map(|i| {
                let group = captures.get(i).unwrap();
                [group.start(), group.end()]
            })
            .collect(),
    };

    assert_eq!(
        matches, expected,
        "wrong captures for /{pattern}/ on '{input}'",
    );
}
