"""Dataset download, caching, and built-in knowledge-base utilities."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

_CACHE_DIR = Path.home() / ".llm_gnns_course" / "data"

_DATASETS: dict[str, dict] = {
    "telco_churn": {
        "url": (
            "https://raw.githubusercontent.com/IBM/"
            "telco-customer-churn-on-icp4d/master/data/"
            "Telco-Customer-Churn.csv"
        ),
        "filename": "telco_churn.csv",
    }
}


def load_telco_churn(force_download: bool = False) -> pd.DataFrame:
    """Download (once) and return the IBM Telco Customer Churn dataset as-is.

    The dataset describes ~7 000 telecommunications customers and whether
    they churned (cancelled their subscription).  It is the running dataset
    for Module I labs.  Cleaning is intentionally left as a lab exercise.

    Parameters
    ----------
    force_download:
        Re-download even if a cached copy already exists.

    Returns
    -------
    pd.DataFrame
        Raw CSV as a DataFrame — no cleaning applied.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta = _DATASETS["telco_churn"]
    path = _CACHE_DIR / meta["filename"]

    if not path.exists() or force_download:
        print(f"Downloading Telco Churn dataset → {path} …", flush=True)
        urllib.request.urlretrieve(meta["url"], path)
        print("Done.", flush=True)
    else:
        print(f"Using cached dataset at {path}", flush=True)

    return pd.read_csv(path)


# ── Module II: TechRetail knowledge base ─────────────────────────────────────

_TECHRETAIL_KB: list[dict] = [
    {
        "id": "doc_001",
        "title": "Return and Refund Policy",
        "content": (
            "TechRetail accepts returns within 30 days of purchase for most products. "
            "Electronics (laptops, tablets, phones, cameras) must be returned within 15 days "
            "and require all original packaging, accessories, and documentation. "
            "Software licences, digital downloads, and personalised items cannot be returned "
            "once activated or delivered. "
            "To start a return, email returns@techretail.example with your order number and reason. "
            "Approved refunds are credited to the original payment method within 5 to 7 business days "
            "after the returned item is received. "
            "Return shipping costs are covered by TechRetail only if the item is defective or the "
            "wrong product was shipped."
        ),
        "source": "Customer Policy Manual v2.4 — Section 4",
    },
    {
        "id": "doc_002",
        "title": "Shipping and Delivery",
        "content": (
            "Standard shipping (5–7 business days) costs 14,900 COP for orders under 150,000 COP. "
            "Orders over 150,000 COP qualify for free standard shipping. "
            "Express shipping (2–3 business days) costs 39,900 COP regardless of order size. "
            "Same-day delivery is available in Bogotá, Medellín, and Cali for orders placed before 10:00 AM. "
            "Same-day delivery costs 24,900 COP and requires a minimum order of 80,000 COP. "
            "International shipping is available to 12 countries in Latin America. "
            "All orders receive a tracking code by email within 24 hours of dispatch."
        ),
        "source": "Customer Policy Manual v2.4 — Section 5",
    },
    {
        "id": "doc_003",
        "title": "TechRetail Premium Membership",
        "content": (
            "TechRetail Premium is an annual subscription at 89,900 COP per year. "
            "Benefits include: free express shipping on all orders, 10% discount on all products, "
            "priority access to sales events 24 hours before the general public, "
            "an extended return window of 60 days (vs. the standard 30 days), "
            "and a dedicated customer support line with no wait times. "
            "Premium membership is per person and non-transferable. "
            "Members can cancel at any time; refunds are prorated for unused months."
        ),
        "source": "Premium Membership Guide v1.3",
    },
    {
        "id": "doc_004",
        "title": "Warranty Information",
        "content": (
            "All TechRetail products include a 1-year manufacturer warranty against defects. "
            "Extended warranties are available for purchase: a 2-year plan at 15% of the product price, "
            "and a 3-year plan at 25% of the product price. "
            "Extended warranties cover accidental damage (drops, liquid spills) in addition to "
            "manufacturing defects. "
            "Warranty claims must be submitted through the TechRetail website within the warranty period. "
            "Processing time for warranty claims is 10–15 business days. "
            "If a product cannot be repaired, TechRetail replaces it with the same or an equivalent model."
        ),
        "source": "Product Warranty Terms v3.1",
    },
    {
        "id": "doc_005",
        "title": "Customer Support Hours and Channels",
        "content": (
            "TechRetail customer support is available through: "
            "Live chat on the website — Monday to Friday, 8:00 AM to 8:00 PM (Colombia time). "
            "Phone support at +57 1 234 5678 — Monday to Saturday, 9:00 AM to 6:00 PM. "
            "Email at soporte@techretail.example — response within 24 business hours. "
            "Premium members have access to a dedicated support line with no wait times. "
            "For order tracking, customers can use the self-service portal 24 hours a day, 7 days a week. "
            "Technical support for electronics is only available via live chat, on Tuesdays and Thursdays "
            "from 10:00 AM to 4:00 PM."
        ),
        "source": "Support Operations Manual v2.0",
    },
    {
        "id": "doc_006",
        "title": "Loyalty Points Programme",
        "content": (
            "Every purchase earns TechRetail Points at the rate of 1 point per 1,000 COP spent. "
            "Points can be redeemed at checkout: 100 points equals a 5,000 COP discount. "
            "Points expire 18 months after they are earned if unused. "
            "Bonus events: double points on the last weekend of every month; "
            "triple points during the annual TechRetail Day sale (first week of November). "
            "Premium members earn 1.5× points on every purchase. "
            "Points cannot be transferred between accounts or exchanged for cash."
        ),
        "source": "Loyalty Programme Rules v4.0",
    },
    {
        "id": "doc_007",
        "title": "Store Locations",
        "content": (
            "TechRetail headquarters: Carrera 15 No. 88-64, Bogotá. Open Monday to Friday, 9 AM to 5 PM. "
            "Showroom Bogotá: Centro Comercial Andino, Local 342. Open daily 10 AM to 8 PM. "
            "Showroom Medellín: El Tesoro Parque Comercial, Local 211. Open daily 10 AM to 8 PM. "
            "Showroom Cali: Unicentro Cali, Local 115. Open Monday to Saturday, 10 AM to 7 PM. "
            "All showrooms offer in-person technical support and warranty claim drop-off. "
            "Online orders can be collected in-store if placed before noon on the same day."
        ),
        "source": "Store Operations Guide v1.2",
    },
    {
        "id": "doc_008",
        "title": "Data Privacy and Cookie Policy",
        "content": (
            "TechRetail collects purchase history, browsing behaviour, and contact information "
            "to personalise recommendations and improve service. "
            "Data is never sold to third parties. "
            "TechRetail shares data only with logistics partners (for delivery) and payment processors. "
            "Customers can request a full export of their data or account deletion by emailing "
            "privacidad@techretail.example. "
            "Requests are processed within 15 business days as required by Colombian data protection law "
            "(Ley 1581 de 2012). "
            "Analytics cookies can be disabled in account settings."
        ),
        "source": "Privacy Policy v3.3",
    },
]


def load_company_kb() -> list[dict]:
    """Return the TechRetail Co. knowledge base used in Module II (RAG labs).

    TechRetail is a fictional Colombian electronics retailer.  The documents
    contain specific policies (prices in COP, deadlines, contacts) that a
    general-purpose LLM cannot know — making them ideal for demonstrating
    when RAG helps vs. when an LLM alone hallucinates.

    Returns
    -------
    list[dict]
        Each document has keys ``id``, ``title``, ``content``, ``source``.
    """
    return [doc.copy() for doc in _TECHRETAIL_KB]
