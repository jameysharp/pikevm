#![no_main]
use libfuzzer_sys::fuzz_target;

use pcre2::bytes::Regex;
use pikevm::compile;

macro_rules! fail_silently {
    ( $e:expr ) => {
        if let Ok(v) = $e { v } else { return; }
    }
}

fuzz_target!(|data: (&str, &str)| {
    let (pattern, input) = data;

    // Require both PCRE and regex-syntax to accept the pattern.
    let regex = fail_silently!(Regex::new(pattern));
    let program = fail_silently!(compile(pattern));

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
