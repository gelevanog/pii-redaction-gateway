"""Playground examples (written for the demo; not part of the gold set) and task prompts."""

from __future__ import annotations

EXAMPLES: list[dict[str, str]] = [
    {
        "id": "support-ticket",
        "label": "Support ticket (EN)",
        "text": (
            "Hi Brightloop team,\n\nmy name is Anna Petrova and my Aura thermostat stopped heating after the update. "
            "You can reach me at anna.petrova@gmail.com or +44 7700 900183. The unit is installed at "
            "14 Elm Grove, Bristol BS6 5NP. Ms. Petrova is my name on the account, but my husband "
            "David Okafor set it up.\n\nThanks,\nAnna"
        ),
    },
    {
        "id": "crm-note",
        "label": "CRM note with payment data",
        "text": (
            "Call with Marcus Lindqvist (CFO, Nordhavn Shipping) on Tuesday. He wants the refund to card "
            "4539 1488 0343 6467, or by bank transfer to SE45 5000 0000 0583 9825 7466. Follow up with his "
            "assistant Priya Raman, priya.raman@nordhavn-shipping.se."
        ),
    },
    {
        "id": "german-email",
        "label": "German email",
        "text": (
            "Sehr geehrte Damen und Herren,\n\nich bin umgezogen. Meine neue Adresse lautet Lindenstraße 27, "
            "50674 Köln. Bitte buchen Sie künftig von DE89 3704 0044 0532 0130 00 ab. Telefonisch erreichen "
            "Sie mich unter 0221 4710 3392.\n\nMit freundlichen Grüßen\nJürgen Hoffmann"
        ),
    },
    {
        "id": "log-secret",
        "label": "Log line with a secret",
        "text": (
            "2026-10-04T09:12:44Z ERROR auth failed user=kim.nguyen@example-corp.com ip=203.0.113.45 "
            "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJraW0ubmd1eWVuIn0.c2lnbmF0dXJlLXZhbHVl "
            "retrying with api_key=sk-demo-9fK2LmQ8xR4tB7vN1cZ6"
        ),
    },
    {
        "id": "hr-note",
        "label": "HR note (DOB, SSN)",
        "text": (
            "New hire: Grace Whitfield, DOB 03/14/1991, SSN 512-44-7809. Start date is November 3. Her manager "
            "Will Ortega asked IT to create the account before Monday; Grace prefers to be called Gracie."
        ),
    },
    {
        "id": "spanish-chat",
        "label": "Spanish chat",
        "text": (
            "Hola, soy Lucía Fernández Ortega, mi DNI es 48291037Q. Vivo en Calle de Alcalá 112, 28009 Madrid "
            "y mi móvil es +34 612 48 93 07. ¿Pueden cambiar la dirección de envío?"
        ),
    },
]

TASKS: dict[str, dict[str, str]] = {
    "reply": {
        "label": "Draft a reply",
        "prompt": (
            "You are a customer-support agent for Brightloop, a smart-home company. Draft a short, friendly reply "
            "to the customer's message. Address the customer by name."
        ),
    },
    "summarize": {
        "label": "Summarize",
        "prompt": (
            "Summarize the text in at most three bullet points for a colleague. Keep the people and contact "
            "details that matter."
        ),
    },
    "actions": {
        "label": "Extract action items",
        "prompt": ("List the action items in the text as bullet points, each with who should do it."),
    },
}
