# Security and privacy notes

ReliTrace v0.4 is an **experimental** data-shift diagnostic app, not a production-approved environment for confidential data.

- Do not upload personally identifiable, confidential, medical, financial account, or regulated data to a public deployment.
- CSV uploads are processed in server memory, subject to application size and schema validation; the hosting platform and application operator still handle data during processing.
- ZIP report import reads only known aggregate report filenames, with size limits; it does not extract or execute archive contents.
- Uploaded features are not automatically anonymized or de-identified.
- No user authentication, persistent encrypted storage, or formal security audit is included.
- Never store tokens or credentials in source control. Use the hosting platform's secret-management facility if credentials are later required.
- File a GitHub issue for non-sensitive bugs. For suspected vulnerabilities, avoid posting exploit details or private data in public issues; contact the repository maintainer privately using an appropriate method configured by them.
