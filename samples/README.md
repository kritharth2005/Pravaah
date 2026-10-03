# Test documents

Fictional legal documents for trying Pravaah end to end. All names, case numbers, IDs and amounts are invented, and every page is marked as a specimen.

| File | Try it with | What it exercises |
|------|-------------|-------------------|
| `01_legal_notice_rent_arrears.pdf` | Citizen → Generate Summary | Short document: single-pass summary of the notice itself (Transfer of Property Act s.106) |
| `02_consumer_commission_order.pdf` | Professional → Generate Summary | Long judgment (~12.9k characters, over the 12k single-prompt limit): summarised in parts, then combined (Consumer Protection Act, 2019) |
| `03_police_complaint_theft.pdf` | Citizen or Professional → Generate Advice | Criminal matter citing BNS ss. 303, 305 and 331 |
| `04_scanned_wage_complaint.pdf` | Either portal | Image-only scan with no text layer: needs Tesseract. Works in the Docker image; on a host without Tesseract the API returns 503 |

On CPU, expect 2–3 minutes per answer, longer for document 02 or with translation.
