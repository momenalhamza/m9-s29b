# Learner Notes — Stretch Thu (GraphRAG hybrid retrieval)

Use this file to capture your design decisions and what you observed.
It is read by the TA during grading.

## Design decisions

### Vector candidates

I called the native procedure
`db.index.vector.queryNodes('recipe_descriptions', $k, $vector)`, binding
both `$k` and the 384-dim query vector through the Bolt parameter map —
never string-interpolated — so the query embedding rides the exact same
MiniLM path (`embed_text`) that `load_fixture.py` used for the stored
embeddings. Keeping query and corpus in one embedding space is what makes
the cosine score meaningful; mixing models would technically still return
384-dim vectors but in an incompatible space.

The one parameter that matters here is the index's `vector.dimensions`
(384) and `similarity_function` (cosine) — both fixed by the fixture, so
I deliberately did *not* tune them; changing the dimension would raise a
mismatch at query time. I considered over-fetching a larger candidate-k
than the returned-k (e.g. fetch 30, fuse, return 10) so the structural
boost has more room to reorder borderline candidates. On this 50-recipe
fixture with k≤10 it made no measurable difference, so I kept
candidate-k == returned-k for simplicity. On a larger corpus I would
revisit this: over-fetching is the cheap lever that lets fusion actually
change the top-k rather than just reshuffle within it.

### Traversal

I fetched context **lazily, one Cypher query per candidate**, exactly as
the assignment allows. With a 50-recipe fixture and k≤10 that is at most
10 single-`MATCH` lookups per query — negligible latency, and the code
stays readable (one query = one well-understood structural hop). The
batched alternative (`UNWIND $ids AS id MATCH (r:Recipe {id:id}) ...`,
one round-trip) is the right call once candidate-k grows or the driver is
remote, because per-candidate queries pay N network round-trips. I noted
this trade-off rather than implementing it, since here the round-trip
cost is dominated by the embedding step anyway.

One detail: `collect(DISTINCT i.name)` over an absent `OPTIONAL MATCH`
returns `[]`, but I still filter out `None` before sorting so the
ingredient list is always a clean, alphabetically-sorted `list[str]` —
deterministic output regardless of the recipe's edge set.

### Fusion

I implemented the documented rule exactly:
`final = vector_score + 0.1 * [cuisine? + author? + ingredients?]`. A
field counts only if non-empty (not `None`, and for ingredients not
`[]`). This keeps the fused score in a comparable range (vector ∈ [0,1]
plus up to 0.3) and demotes the fixture's deliberately context-bare
recipes below context-rich ones whose vector scores are close.

**Where this rule under-performs:** the boost is a flat
*presence* indicator — it rewards *having* a cuisine/author/ingredients
at all, but is blind to *which* ones. Two recipes that both have all
three fields get the identical +0.3, even if one is a perfect cuisine
match for the query and the other is off-topic. So the rule helps most
on the bare-vs-rich distinction and not at all on rich-vs-rich ranking.

**Alternative rule:** weight the structural term by *relevance* rather
than presence — e.g. an ingredient-overlap Jaccard between query terms
and the recipe's ingredient set, plus a hierarchical-cuisine bonus when
the recipe's cuisine is the query's cuisine or an ancestor via
`(start)-[:SUBCLASS_OF*0..]->(ancestor)` (recall the edge direction is
child→parent, so Sichuan→Chinese). Normalized into [0,1] and added with
a tuned weight, this would most help the **Sichuan** and **lasagna**
queries: Sichuan because the cuisine hierarchy lets a "Sichuan" query
also match recipes tagged only as the parent "Chinese"; lasagna because
ingredient overlap (pasta, béchamel, ground meat) would separate the
true baked-pasta golds from lexically-similar-but-distinct pasta dishes.

## Eval observations

Measured macro recall@10 = **0.812** (per-query, macro-averaged over the
8 queries; well above the ≥0.6 threshold).

- **Retrieved well (recall 1.00):** the Sichuan stir-fried chicken, the
  numbing-sauce tofu, the Caprese (tomato/mozzarella/basil), the
  sushi-roll query, and the tonkotsu noodle-soup query. These pair
  strong sensory/ingredient descriptors ("peppercorn", "numbing",
  "seaweed", "pork broth") with descriptions that lexically and
  semantically echo them — MiniLM nails the embedding and graph context
  only confirms the ranking.

- **Surfaced fewer golds:**
  - *creamy oven-baked layered pasta with meat sauce* (recall 0.33) —
    the weakest. The failure is **lexically similar but topically
    distant**: MiniLM pulls in other creamy/cheesy pasta dishes that
    aren't the gold lasagna set. The vector space can't tell "layered
    baked" apart from "creamy pasta" generally.
  - *fluffy stack of pancakes ... for breakfast* (recall 0.50) — a
    category/breakfast query where the descriptor "fluffy stack" is
    generic; semantically near several breakfast items, diluting the
    golds.
  - *warm tortillas with marinated grilled meat and lime* (recall 0.67) —
    near-miss; one gold taco/fajita variant sits just outside top-10
    behind topically-adjacent Mexican dishes.

  In every weak case hybrid recall == vector recall, which is the
  telling part: when the *misses* are themselves context-rich recipes,
  the flat +0.3 presence boost can't break the tie — confirming the
  fusion-rule limitation above and motivating the relevance-weighted
  alternative.

- **On MiniLM-L6-v2 as the embedding choice:** it's an excellent,
  cheap default for short recipe descriptions and clearly carries most
  of the recall here. Its limit shows on *category/structure* queries
  ("layered baked", "breakfast stack") where the distinguishing signal
  is compositional rather than lexical — a small 384-dim general encoder
  collapses those into a nearby semantic blob. That's exactly the gap
  the graph side is meant to close, and why a *relevance-aware* fusion
  (ingredient overlap, cuisine hierarchy) would lift the weak queries
  more than the presence-only rule can.

## Production framing

GraphRAG hybrid retrieval — vector index + graph traversal fused at
answer time — is a first-class production pattern; it is the same
*schema-bounded NL→evidence* primitive you saw on Weaviate in Module 8
and will see again on the deployment surface in Module 10. The engine
differs (Weaviate's HNSW versus Neo4j's native vector index); the
primitive does not.

One result the Neo4j-side fusion gives that a vector-only Weaviate
pipeline could not: a query like *"fragrant Sichuan noodle dishes"* can
surface recipes tagged only as the parent **Chinese** cuisine by walking
`[:SUBCLASS_OF*0..]` from Sichuan up the taxonomy, and can demote
context-bare recipes that embed similarly but carry no graph evidence.
Pure vector search has no notion of "is Sichuan a kind of Chinese" or
"does this node actually have cuisine/author/ingredient edges" — that
relational evidence lives in the graph, and fusing it at answer time is
precisely what lifts retrieval beyond nearest-neighbour-in-embedding.
