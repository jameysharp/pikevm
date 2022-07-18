# pikevm, an implementation of "Regular Expression Matching: the Virtual Machine Approach"

In 2009, Russ Cox wrote a series of blog posts about ways to implement
regular expressions. The second post in the series was ["Regular
Expression Matching: the Virtual Machine Approach"][vm].

[vm]: https://swtch.com/~rsc/regexp/regexp2.html

This repo contains a Rust implementation of the section of that post
titled "Pike's Implementation", together with the enhancements from the
section titled "Ambiguous Submatching". This implementation is not
intended for production use, but rather to aid in understanding the blog
post.

The code is largely uncommented; see Cox's blog posts for most of the
explanation of what's going on. That said, there are some details that
Cox largely handwaved which I've made explicit in this implementation,
so I hope reading the code might also help in understanding the blog
post.

The implementation is pretty simple: Compiling a regex is around 170
lines of code, and matching is about 100 lines. I've put no real effort
into performance, just aiming for a (hopefully) readable implementation.

Like the blog post, this version supports capture groups with both
greedy and non-greedy repetition operators. I've also extended the
original algorithm to match a set of regexes against a single input all
at once.

This implementation only supports patterns which are anchored at both
ends, but you can insert `.*?` at the beginning or end of the pattern to
get unanchored matches. Similarly, the conventional capture group `$0`
isn't provided automatically; wrap the entire pattern in a capture if
you want it.

For a given regex, this algorithm uses O(1) space, and O(n) time in the
length of the input; asymptotically it should be identical to a DFA
implementation. The constant factors are probably different, of course,
and this doesn't do automaton minimization or other optimization.

I believe the algorithm in [A Play on Regular Expressions][play] (one of
my favorite papers!) could also do all this, given a careful choice of
semiring for the weights. In that algorithm, weights are moved around
the structure of the regular expression in roughly the same way that the
VM approach uses its "threads". The tricky part is prioritization when
combining weights from ambiguous submatches, I think, but the solutions
for the VM approach should transfer pretty cleanly.

[play]: https://sebfisch.github.io/haskell-regexp/

(That paper describes a Haskell implementation, but I've previously
written a [Rust weighted regular expressions][weighted-regexp-rs]
library as well.)

[weighted-regexp-rs]: https://github.com/jameysharp/weighted-regexp-rs

I've often found it helpful to switch between [NFA][], virtual machine,
and weighted-regexp interpretations when I'm trying to reason about
regular expressions. My hope is that pulling together these
superficially different perspectives will be useful to other people too.

[NFA]: https://en.wikipedia.org/wiki/Nondeterministic_finite_automaton
