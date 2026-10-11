"""Generate the evaluation set: six policy documents of a fictional company with
12 planted contradictions, plus near-misses that are NOT contradictions.

Run: python evaluation/make_test_docs.py
Writes evaluation/docs/*.pdf and evaluation/ground_truth.json
"""

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

HERE = Path(__file__).resolve().parent

# Each document is a list of pages; each page is (heading, text).
DOCUMENTS = {
    "employee_handbook.pdf": [
        (
            "Working time and leave",
            "Halden Freight employees work 40 hours per week. Core hours are from 10:00 to 15:00 on "
            "working days, and every employee is expected to be reachable during this time. Full-time "
            "employees receive 28 days of paid vacation per year. Employees with more than 10 years of "
            "service receive 32 days of paid vacation per year. Vacation requests must be submitted at "
            "least 2 weeks in advance through the HR portal. An employee who is ill for more than 3 "
            "consecutive days must provide a medical certificate.",
        ),
        (
            "Probation and remote work",
            "The probation period for new employees lasts 6 months. During probation, either side may "
            "end the employment with 2 weeks of notice. After probation the notice period is 3 months. "
            "Employees may work remotely up to 3 days per week, provided their manager agrees. Interns "
            "may work remotely 1 day per week.",
        ),
        (
            "Expenses and conduct",
            "Expense claims must be submitted within 14 days of the end of a business trip. All expense "
            "claims are approved by the Finance team before reimbursement. Reimbursement is paid with "
            "the next monthly salary. Gifts from suppliers worth more than 50 euros must be declined or "
            "reported to the Compliance office. The annual performance review takes place in March.",
        ),
    ],
    "remote_work_guideline.pdf": [
        (
            "Remote work rules",
            "This guideline applies to all employees of Halden Freight who work outside the office. "
            "Employees may work remotely up to 2 days per week. Interns may work remotely 1 day per "
            "week. When working remotely, employees must be reachable during core hours, which are "
            "from 9:00 to 16:00 on working days. Remote work from outside the European Union requires "
            "written approval from HR at least 4 weeks in advance.",
        ),
        (
            "Security when working remotely",
            "Remote access to company systems is only allowed through the company VPN with two-factor "
            "authentication. Passwords for company accounts must be changed every 180 days. A lost or "
            "stolen company laptop must be reported to the IT service desk within 72 hours. Company "
            "devices must not be used by family members. Employees receive a monitor and a keyboard "
            "for their home office on request.",
        ),
    ],
    "it_security_policy.pdf": [
        (
            "Accounts and passwords",
            "Passwords for company accounts must be at least 14 characters long. Passwords for "
            "administrator accounts must be at least 20 characters long. Passwords for company accounts "
            "must be changed every 90 days. Two-factor authentication is mandatory for remote access "
            "through the company VPN. Accounts are locked after 5 failed login attempts.",
        ),
        (
            "Devices and data",
            "A lost or stolen company laptop must be reported to the IT service desk within 24 hours. "
            "All company laptops use full disk encryption. File servers are backed up daily, and "
            "backups are stored in a second data centre. Customer records are deleted 2 years after the "
            "end of the contract. Security awareness training is mandatory once per year.",
        ),
    ],
    "travel_expense_policy.pdf": [
        (
            "Booking",
            "Business trips must be approved by the line manager before booking. Train travel is booked "
            "in second class; first class is allowed for journeys longer than 4 hours. Hotel costs are "
            "reimbursed up to 150 euros per night. In London and Zurich, hotel costs are reimbursed up "
            "to 220 euros per night. Flights within Europe are booked in economy class.",
        ),
        (
            "Claims",
            "Expense claims must be submitted within 30 days of the end of a business trip. All expense "
            "claims are approved by the department head before reimbursement. Meals are reimbursed up "
            "to 40 euros per day. Receipts are required for every expense above 10 euros. Private use "
            "of a rental car is not reimbursed.",
        ),
    ],
    "data_retention_policy.pdf": [
        (
            "Retention periods",
            "This policy defines how long Halden Freight keeps records. Customer records are retained "
            "for 5 years after the end of the contract. Invoices and accounting records are retained "
            "for 10 years. Application documents of rejected applicants are deleted after 6 months. "
            "Personnel files are retained for 3 years after an employee leaves.",
        ),
        (
            "Backups and deletion",
            "File servers are backed up weekly, and backups are stored in a second data centre. "
            "Backups are kept for 90 days. Data is deleted using certified erasure software. Deletion "
            "requests from customers are answered within 30 days. The Data Protection Officer reviews "
            "this policy once per year.",
        ),
    ],
    "onboarding_guide.pdf": [
        (
            "Welcome",
            "Welcome to Halden Freight. Your probation period lasts 3 months. As a full-time employee "
            "you receive 30 days of paid vacation per year. If you are ill for more than 3 consecutive "
            "days, you must provide a medical certificate. Your first feedback meeting with your "
            "manager takes place after 6 weeks.",
        ),
        (
            "Getting set up",
            "Your password for company accounts must be at least 10 characters long. Two-factor "
            "authentication is mandatory for remote access through the company VPN. On business trips, "
            "hotel costs are reimbursed up to 120 euros per night. Security awareness training is "
            "mandatory once per year. Your laptop is handed out by the IT service desk on your first day.",
        ),
    ],
}

# The planted contradictions. "values" lists accepted spellings of each side's value;
# evaluate.py uses them to recognise a reported conflict.
GROUND_TRUTH = [
    {"id": "vacation_days", "a": {"source": "employee_handbook.pdf", "values": ["28"]},
     "b": {"source": "onboarding_guide.pdf", "values": ["30"]}},
    {"id": "remote_days", "a": {"source": "employee_handbook.pdf", "values": ["3 days", "three days"]},
     "b": {"source": "remote_work_guideline.pdf", "values": ["2 days", "two days"]}},
    {"id": "probation_period", "a": {"source": "employee_handbook.pdf", "values": ["6 months", "six months"]},
     "b": {"source": "onboarding_guide.pdf", "values": ["3 months", "three months"]}},
    {"id": "password_length", "a": {"source": "it_security_policy.pdf", "values": ["14"]},
     "b": {"source": "onboarding_guide.pdf", "values": ["10"]}},
    {"id": "password_rotation", "a": {"source": "it_security_policy.pdf", "values": ["90 days"]},
     "b": {"source": "remote_work_guideline.pdf", "values": ["180 days"]}},
    {"id": "expense_deadline", "a": {"source": "travel_expense_policy.pdf", "values": ["30 days"]},
     "b": {"source": "employee_handbook.pdf", "values": ["14 days"]}},
    {"id": "hotel_limit", "a": {"source": "travel_expense_policy.pdf", "values": ["150"]},
     "b": {"source": "onboarding_guide.pdf", "values": ["120"]}},
    {"id": "customer_records", "a": {"source": "data_retention_policy.pdf", "values": ["5 years", "five years"]},
     "b": {"source": "it_security_policy.pdf", "values": ["2 years", "two years"]}},
    {"id": "lost_laptop", "a": {"source": "it_security_policy.pdf", "values": ["24 hours"]},
     "b": {"source": "remote_work_guideline.pdf", "values": ["72 hours"]}},
    {"id": "expense_approval", "a": {"source": "travel_expense_policy.pdf", "values": ["department head"]},
     "b": {"source": "employee_handbook.pdf", "values": ["finance team"]}},
    {"id": "backup_frequency", "a": {"source": "it_security_policy.pdf", "values": ["daily"]},
     "b": {"source": "data_retention_policy.pdf", "values": ["weekly"]}},
    {"id": "core_hours", "a": {"source": "employee_handbook.pdf", "values": ["10:00"]},
     "b": {"source": "remote_work_guideline.pdf", "values": ["9:00"]}},
]


def write_pdf(path: Path, pages: list[tuple[str, str]]) -> None:
    styles = getSampleStyleSheet()
    story = []
    for number, (heading, text) in enumerate(pages):
        if number:
            story.append(PageBreak())
        story += [Paragraph(heading, styles["Heading2"]), Spacer(1, 12), Paragraph(text, styles["BodyText"])]
    SimpleDocTemplate(str(path), pagesize=A4, title=path.stem).build(story)


def main() -> None:
    docs_dir = HERE / "docs"
    docs_dir.mkdir(exist_ok=True)
    for name, pages in DOCUMENTS.items():
        write_pdf(docs_dir / name, pages)
    (HERE / "ground_truth.json").write_text(json.dumps(GROUND_TRUTH, indent=2))
    print(f"Wrote {len(DOCUMENTS)} PDFs and {len(GROUND_TRUTH)} planted contradictions to {HERE}")


if __name__ == "__main__":
    main()
