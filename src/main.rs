use pcre2::bytes::Regex;
use pikevm::compile;

fn main() -> Result<(), pikevm::Error> {
    env_logger::init();

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let (input, patterns) = args.split_last().unwrap();
    let compiled = patterns
        .iter()
        .map(|pattern| compile(&pattern))
        .collect::<Result<Vec<_>, _>>()?;
    let input_bytes = input.as_bytes();

    for (pattern, program) in patterns.iter().zip(compiled) {
        println!("pattern '{}', input '{}'", pattern, input);

        print!("Pike VM: ");
        report(input, program.exec(input_bytes).map(groups));

        let dfa = program.to_dfa();
        log::info!("DFA for /{}/:\n{}", pattern, dfa.to_dot());
        print!("DFA: ");
        report(input, dfa.exec(input_bytes).map(groups));

        print!("PCRE: ");
        let regex = Regex::new(pattern).unwrap();
        let captures = regex.captures(input.as_bytes()).unwrap();
        report(
            input,
            captures.map(|captures| {
                (0..captures.len())
                    .map(|i| captures.get(i).map(|group| (group.start(), group.end())))
                    .collect()
            }),
        );
    }
    Ok(())
}

fn groups(captures: Vec<usize>) -> Vec<Option<(usize, usize)>> {
    captures
        .chunks_exact(2)
        .map(|group| {
            debug_assert_eq!(group[0] == usize::MAX, group[1] == usize::MAX);
            if group[0] == usize::MAX {
                None
            } else {
                Some((group[0], group[1]))
            }
        })
        .collect()
}

fn report(input: &str, captures: Option<Vec<Option<(usize, usize)>>>) {
    if let Some(captures) = captures {
        println!("MATCH");
        for (idx, group) in captures.into_iter().enumerate() {
            print!(" -- {}: ", idx);
            if let Some((start, end)) = group {
                println!(
                    "\"{}\" ({},{})",
                    String::from_utf8_lossy(&input.as_bytes()[start..end]),
                    start,
                    end
                );
            } else {
                println!("no match");
            }
        }
    } else {
        println!("NO MATCH");
    }
}
