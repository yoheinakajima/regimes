"""Pinned stoplist for the lexical signal.

Vocabulary frozen at construction. Re-ingest of the same corpus must
produce a byte-identical event log; mutating this list shifts the
lexical signal across the entire benchmark.

Sourced from the LME reference spec (activegraph_lme/activegraph/stoplist.py)
to keep graph content matched to the reference. The set is identical.
"""

from __future__ import annotations

STOPLIST: frozenset[str] = frozenset(
    {
        "about", "above", "after", "again", "against", "all", "also", "always",
        "another", "any", "anyone", "anything", "are", "aren", "around", "back",
        "because", "been", "before", "being", "below", "between", "both", "but",
        "came", "can", "cannot", "could", "couldn", "did", "didn", "does", "doesn",
        "doing", "don", "done", "down", "during", "each", "either", "else", "even",
        "ever", "every", "feel", "felt", "for", "from", "get", "gets", "getting",
        "give", "given", "gives", "going", "gone", "got", "had", "hadn", "has",
        "hasn", "have", "haven", "having", "her", "here", "hers", "herself", "him",
        "himself", "his", "how", "however", "into", "isn", "its", "itself", "just",
        "knew", "know", "known", "knows", "least", "less", "like", "liked", "likes",
        "look", "looked", "looking", "looks", "made", "make", "makes", "making",
        "many", "may", "maybe", "might", "more", "most", "much", "must", "myself",
        "need", "needed", "needs", "never", "next", "not", "nothing", "now", "off",
        "often", "okay", "once", "one", "only", "other", "others", "our", "ours",
        "ourselves", "out", "over", "own", "perhaps", "really", "right", "same",
        "saw", "say", "saying", "says", "see", "seeing", "seems", "seen", "shall",
        "shan", "she", "should", "shouldn", "since", "some", "someone", "something",
        "sometimes", "soon", "still", "such", "take", "taken", "takes", "taking",
        "tell", "telling", "tells", "than", "thank", "thanks", "that", "the", "their",
        "theirs", "them", "themselves", "then", "there", "these", "they", "thing",
        "things", "think", "thinking", "thinks", "this", "those", "though", "through",
        "thus", "tried", "tries", "true", "try", "trying", "under", "until", "very",
        "want", "wanted", "wants", "was", "wasn", "way", "well", "went", "were",
        "weren", "what", "whatever", "when", "where", "whether", "which", "while",
        "who", "whom", "whose", "why", "will", "with", "within", "without", "won",
        "would", "wouldn", "yes", "yet", "you", "your", "yours", "yourself",
        "yourselves",
        "user", "assistant", "session",
    }
)
