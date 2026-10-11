"""Offline stand-ins for the embedding model and the LLM.

They let the whole graph run without network access or an API key. They are
crude on purpose: they check that the plumbing works, not how well a real
model performs.
"""

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

from agent import Claim, Claims, Judgement, Verdict, Verification

STOPWORDS = set(
    "a an the of to in on for and or is are be must may with within by at from as per up "
    "your you our their this that which who it its all every each not than more".split()
)


class WordHashEmbeddings(Embeddings):
    """Texts sharing words get similar vectors, so retrieval is meaningful."""

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * 512
        for word in re.findall(r"[a-z]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 512] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip()) if s]


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower())) - STOPWORDS


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?::\d+)?", text))


def similarity(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def tag_contents(text: str, tag: str) -> list[str]:
    return [m.strip() for m in re.findall(rf"<{tag}[^>]*>\n(.*?)\n</{tag}>", text, flags=re.S)]


class RuleBasedLLM:
    """Pretends to be a chat model with structured output, using word overlap:
    two sentences 'conflict' if they share most words but contain different numbers."""

    judge_threshold = 0.5
    verify_threshold = 0.65

    def with_structured_output(self, schema):
        handler = {Claims: self._extract, Judgement: self._judge, Verification: self._verify}[schema]
        return RunnableLambda(lambda messages: handler(messages[-1][1]))

    def _extract(self, prompt: str) -> Claims:
        passage = tag_contents(prompt, "passage")[0]
        return Claims(claims=[Claim(statement=s) for s in sentences(passage) if numbers(s)])

    def _judge(self, prompt: str) -> Judgement:
        claim = tag_contents(prompt, "claim")[0]
        verdicts = []
        for number, passage in enumerate(tag_contents(prompt, "passage"), start=1):
            best = max(sentences(passage), key=lambda s: similarity(claim, s))
            if similarity(claim, best) < self.judge_threshold:
                relation = "unrelated"
            else:
                relation = "conflict" if numbers(best) != numbers(claim) else "agree"
            verdicts.append(
                Verdict(passage=number, relation=relation, quote="" if relation == "unrelated" else best,
                        explanation=f"Claim: {claim} Passage: {best}")
            )
        return Judgement(verdicts=verdicts)

    def _verify(self, prompt: str) -> Verification:
        a, b = tag_contents(prompt, "passage_a")[0], tag_contents(prompt, "passage_b")[0]
        pairs = [(sa, sb) for sa in sentences(a) for sb in sentences(b) if numbers(sa) != numbers(sb)]
        # Check the pair the report was about: the one most similar to the reported explanation.
        report = prompt.split("<passage_a")[0]
        quote_a, quote_b = max(pairs, key=lambda p: similarity(p[0], report) + similarity(p[1], report))
        confirmed = similarity(quote_a, quote_b) >= self.verify_threshold
        return Verification(quote_a=quote_a, quote_b=quote_b, is_conflict=confirmed,
                            reason="Same subject, different values." if confirmed else "Different subject or scope.")


class ScriptedLLM:
    """Returns whatever the given functions return, for testing specific paths."""

    def __init__(self, extract, judge, verify):
        self.handlers = {Claims: extract, Judgement: judge, Verification: verify}

    def with_structured_output(self, schema):
        handler = self.handlers[schema]
        return RunnableLambda(lambda messages: handler(messages[-1][1]))
