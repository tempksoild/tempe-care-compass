-- Tempe Care Compass — provider curation pipeline
-- Source: Affine NPPES Provider Data (Snowflake Marketplace)
-- Tables: DIM_PROVIDER, DIM_PROVIDER_ADDRESS, DIM_PROVIDER_TAXONOMY, REF_TAXONOMY_CODE

CREATE DATABASE IF NOT EXISTS CARE_AI;
CREATE SCHEMA IF NOT EXISTS CARE_AI.CURATED;

CREATE OR REPLACE DYNAMIC TABLE CARE_AI.CURATED.TEMPE_PROVIDERS
TARGET_LAG = '1 day'
WAREHOUSE = COMPUTE_WH
AS
SELECT DISTINCT
    p.NPI,
    INITCAP(COALESCE(p.ORGANIZATION_NAME_LBN, p.PROVIDER_FULL_NAME)) AS name,
    p.ENTITY_TYPE_CODE,
    CASE
        WHEN r.TAXONOMY_CLASSIFICATION ILIKE '%pharmacy%' OR r.TAXONOMY_TYPE ILIKE '%pharmacy%' THEN 'Pharmacy'
        WHEN r.TAXONOMY_TYPE ILIKE '%dental%' OR r.TAXONOMY_CLASSIFICATION ILIKE '%dentist%' THEN 'Dental'
        WHEN r.TAXONOMY_TYPE ILIKE '%behavioral%' OR r.TAXONOMY_CLASSIFICATION ILIKE '%counselor%'
             OR r.TAXONOMY_CLASSIFICATION ILIKE '%social worker%' OR r.TAXONOMY_CLASSIFICATION ILIKE '%psycholog%'
             OR r.TAXONOMY_SPECIALIZATION ILIKE '%mental health%' THEN 'Behavioral health'
        WHEN r.TAXONOMY_CLASSIFICATION ILIKE '%radiology%' OR r.TAXONOMY_CLASSIFICATION ILIKE '%radiolog%'
             OR r.TAXONOMY_SPECIALIZATION ILIKE '%imaging%' THEN 'Imaging'
        WHEN r.TAXONOMY_CLASSIFICATION ILIKE '%hospital%' OR r.TAXONOMY_TYPE ILIKE '%hospital%' THEN 'Hospital'
        WHEN r.TAXONOMY_CLASSIFICATION ILIKE '%clinic%' OR r.TAXONOMY_CLASSIFICATION ILIKE '%family medicine%'
             OR r.TAXONOMY_CLASSIFICATION ILIKE '%internal medicine%' OR r.TAXONOMY_CLASSIFICATION ILIKE '%nurse practitioner%'
             OR r.TAXONOMY_CLASSIFICATION ILIKE '%physician assistant%' OR r.TAXONOMY_CLASSIFICATION ILIKE '%general practice%'
             OR r.TAXONOMY_SPECIALIZATION ILIKE '%family%' OR r.TAXONOMY_SPECIALIZATION ILIKE '%primary care%' THEN 'Clinic / primary care'
        ELSE 'Other care'
    END AS category,
    COALESCE(r.TAXONOMY_SPECIALIZATION, r.TAXONOMY_CLASSIFICATION, r.TAXONOMY_TYPE) AS specialty,
    CONCAT_WS(', ', a.ADDRESS_LINE_1, a.ADDRESS_LINE_2) AS address,
    INITCAP(a.CITY_NAME) AS city,
    a.STATE_CODE AS state,
    a.POSTAL_CODE_5 AS zip,
    a.TELEPHONE_NUMBER AS phone,
    p.LAST_UPDATE_DATE AS last_updated,
    'Not stated in NPPES — call to verify' AS affordability,
    'CMS NPPES via Affine' AS source
FROM AFFINE_NPPES_PROVIDER_DATA.REF_DW.DIM_PROVIDER p
JOIN AFFINE_NPPES_PROVIDER_DATA.REF_DW.DIM_PROVIDER_ADDRESS a
    ON a.NPI = p.NPI
JOIN AFFINE_NPPES_PROVIDER_DATA.REF_DW.DIM_PROVIDER_TAXONOMY t
    ON t.NPI = p.NPI AND t.IS_PRIMARY_TAXONOMY = TRUE
JOIN AFFINE_NPPES_PROVIDER_DATA.REF_DW.REF_TAXONOMY_CODE r
    ON r.TAXONOMY_CODE = t.TAXONOMY_CODE
WHERE p.IS_ACTIVE = TRUE
    AND a.ADDRESS_TYPE_CODE = 'PRACTICE'
    AND a.STATE_CODE = 'AZ'
    AND UPPER(a.CITY_NAME) = 'TEMPE';

-- Verified access programs (manually curated — NPPES cannot establish affordability)
CREATE TABLE IF NOT EXISTS CARE_AI.CURATED.VERIFIED_ACCESS_PROGRAMS (
    npi VARCHAR,
    location_address VARCHAR,
    program_type VARCHAR,
    eligibility_notes VARCHAR,
    verification_url VARCHAR,
    verified_at TIMESTAMP_TZ,
    verified_by VARCHAR,
    PRIMARY KEY (npi, location_address)
);

-- Full care options view joining providers with verified programs
CREATE OR REPLACE VIEW CARE_AI.CURATED.TEMPE_CARE_OPTIONS AS
SELECT
    p.*,
    COALESCE(v.program_type, p.affordability) AS affordability_status,
    v.eligibility_notes,
    v.verification_url,
    v.verified_at
FROM CARE_AI.CURATED.TEMPE_PROVIDERS p
LEFT JOIN CARE_AI.CURATED.VERIFIED_ACCESS_PROGRAMS v
    ON p.npi = v.npi AND p.address = v.location_address;

-- Hospital price transparency view (Healthparse Marketplace sample)
CREATE OR REPLACE VIEW CARE_AI.CURATED.HOSPITAL_PRICES AS
SELECT
    "ccn" AS ccn,
    "billing_code" AS billing_code,
    "billing_code_type" AS billing_code_type,
    "billing_code_description" AS billing_code_description,
    "payer_name_canonical" AS payer_name,
    "plan_name" AS plan_name,
    "setting" AS setting,
    "rate_type" AS rate_type,
    "rate_amount" AS rate_amount,
    "rate_methodology" AS rate_methodology,
    "snapshot_date" AS snapshot_date
FROM HEALTHPARSE_HOSPITAL_PRICE_TRANSPARENCY_RATES.HPT_NEGOTIATED_RATES_SAMPLE.HPT_NEGOTIATED_RATES
WHERE "rate_amount" > 0;

-- Price comparison: same procedure across hospitals
CREATE OR REPLACE VIEW CARE_AI.CURATED.PRICE_COMPARISON AS
SELECT
    billing_code,
    billing_code_type,
    billing_code_description,
    COUNT(DISTINCT ccn) AS hospital_count,
    ROUND(MIN(rate_amount), 2) AS min_price,
    ROUND(MAX(rate_amount), 2) AS max_price,
    ROUND(AVG(rate_amount), 2) AS avg_price,
    ROUND(MAX(rate_amount) - MIN(rate_amount), 2) AS price_spread,
    MIN(CASE WHEN rate_type = 'cash' THEN rate_amount END) AS min_cash_price,
    MAX(CASE WHEN rate_type = 'cash' THEN rate_amount END) AS max_cash_price
FROM CARE_AI.CURATED.HOSPITAL_PRICES
GROUP BY billing_code, billing_code_type, billing_code_description;
