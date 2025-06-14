from uuid import uuid4
from typing import List, Dict, Any

def build_transcript_entity(
    channel_id: str,
    transcript_text: str,
    confidence: float,
    is_final: bool,
    offset: str = "PT0S",
    duration: str = "PT1S",
    language: str = "en-US"
) -> Dict[str, Any]:
    return {
        "type": "transcript",
        "id": str(uuid4()),
        "channelId": channel_id,
        "isFinal": is_final,
        "offset": offset,
        "duration": duration,
        "alternatives": [
            {
                "confidence": confidence,
                "languages": [language],
                "interpretations": [
                    {
                        "type": "display",
                        "transcript": transcript_text,
                        "tokens": [
                            {
                                "type": "word",
                                "value": word,
                                "confidence": confidence,
                                "offset": offset,
                                "duration": duration,
                                "language": language
                            }
                            for word in transcript_text.split()
                        ]
                    }
                ]
            }
        ]
    }


def build_agent_assist_entity(
    utterances: List[Dict[str, Any]] = [],
    suggestions: List[Dict[str, Any]] = []
) -> Dict[str, Any]:
    return {
        "type": "agentassist",
        "id": str(uuid4()),
        "utterances": utterances,
        "suggestions": suggestions,
    }


def build_agent_assist_utterance(
    position: str,
    text: str,
    language: str,
    confidence: float,
    channel: str,
    is_final: bool,
    duration: str = "PT1S"
) -> Dict[str, Any]:
    return {
        "id": str(uuid4()),
        "position": position,
        "duration": duration,
        "text": text,
        "language": language,
        "confidence": confidence,
        "channel": channel,
        "isFinal": is_final,
    }


def build_faq_suggestion(
    question: str,
    answer: str,
    confidence: float,
    position: str = "PT0S"
) -> Dict[str, Any]:
    return {
        "type": "faq",
        "id": str(uuid4()),
        "question": question,
        "answer": answer,
        "confidence": confidence,
        "position": position,
    }


def build_article_suggestion(
    title: str,
    excerpts: List[str],
    document_uri: str,
    confidence: float,
    metadata: Dict[str, str] = {},
    position: str = "PT0S"
) -> Dict[str, Any]:
    return {
        "type": "article",
        "id": str(uuid4()),
        "title": title,
        "excerpts": excerpts,
        "documentUri": document_uri,
        "metadata": metadata,
        "confidence": confidence,
        "position": position,
    }
