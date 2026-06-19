"""Vector candidate retrieval over the Neo4j 5.x native vector index.

The vector index `recipe_descriptions` was created and populated by
`load_fixture.py`. Your job here is to query it for the top-k recipes
whose description embedding is most similar to the embedded query.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from .embed import embed_text


def vector_candidates(
    driver,
    embedder: SentenceTransformer,
    query: str,
    k: int = 10,
) -> list[dict]:
    """Return the top-k vector-similar recipes for a natural-language query.

    Returns a list of dicts, each with keys:
      - "recipe_id" (str): the :Recipe.id property
      - "name" (str): the :Recipe.name property
      - "score" (float): similarity score in [0.0, 1.0] (cosine, normalized
        by Neo4j); higher = more similar
    Length of the returned list is at most k. Results are ordered by score DESC.
    """
    qvec = embed_text(embedder, query)

    cypher = """
    CALL db.index.vector.queryNodes('recipe_descriptions', $k, $vector)
    YIELD node, score
    RETURN node.id AS recipe_id, node.name AS name, score
    ORDER BY score DESC
    """

    with driver.session() as session:
        result = session.run(cypher, k=k, vector=qvec)
        return [
            {
                "recipe_id": record["recipe_id"],
                "name": record["name"],
                "score": record["score"],
            }
            for record in result
        ]
