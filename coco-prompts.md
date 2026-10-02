# CoCo runbook

## Discover
`I installed Snowflake Public Data (Free). Find the NPPES tables and explain their grain, joins, active-status logic, address types, taxonomy hierarchy, and fields useful for a Tempe care finder. Cite actual objects and columns; do not guess.`

## Plan
Enter Plan Mode, then: `Use $tempe-care-data. Review sql/00_discover_nppes.sql and sql/01_curate_providers.sql. Correct placeholders and wrong fields from the live catalog. Keep affordability and accepting-patient status unknown because NPPES cannot establish them. Show the plan and wait for approval.`

## Build and test
`Execute the approved plan with an X-Small warehouse. Update the checked-in SQL, run sql/02_quality_checks.sql, diagnose failures, and save query IDs and row counts to docs/coco-run.md.`

## App review
`Review app.py, repository.py, and agent.py. Test Snowflake mode and verify parameter binding, grounding, emergency messaging, and refusal to invent prices or free-clinic status.`
