{{
    config(
        materialized = 'view',
        tags          = ['raw', 'thyroid', 'ingestion']
    )
}}

/*
    Model  : 01_raw_thyroid
    Layer  : Raw / Staging
    Purpose: Expose the source thyroid CSV as a typed, metadata-enriched view.
             Downstream models should reference this via ref('01_raw_thyroid').

    Source config (dbt/sources.yml):
        - name       : synthgen
          tables     :
            - name   : thyroid_clean
              description: Processed CSV produced by src/01_ingest.py
*/

with

source as (

    select * from {{ source('synthgen', 'thyroid_clean') }}

),

staged as (

    select
        -- ── Identifiers ─────────────────────────────────────────────────
        {{ dbt_utils.generate_surrogate_key(['age', 'sex', 'TSH', 'T3', 'TT4']) }}
            as record_id,

        -- ── Demographics ────────────────────────────────────────────────
        cast(age  as float) as age,
        cast(sex  as varchar) as sex,

        -- ── Binary flags (stored as 'f'/'t' in UCI dataset) ─────────────
        cast(on_thyroxine              as boolean) as on_thyroxine,
        cast(query_on_thyroxine        as boolean) as query_on_thyroxine,
        cast(on_antithyroid_medication as boolean) as on_antithyroid_medication,
        cast(sick                      as boolean) as sick,
        cast(pregnant                  as boolean) as pregnant,
        cast(thyroid_surgery           as boolean) as thyroid_surgery,
        cast(I131_treatment            as boolean) as i131_treatment,
        cast(query_hypothyroid         as boolean) as query_hypothyroid,
        cast(query_hyperthyroid        as boolean) as query_hyperthyroid,
        cast(lithium                   as boolean) as lithium,
        cast(goitre                    as boolean) as goitre,
        cast(tumor                     as boolean) as tumor,
        cast(hypopituitary             as boolean) as hypopituitary,
        cast(psych                     as boolean) as psych,

        -- ── Measurement availability flags ──────────────────────────────
        cast(TSH_measured as boolean) as tsh_measured,
        cast(T3_measured  as boolean) as t3_measured,
        cast(TT4_measured as boolean) as tt4_measured,
        cast(T4U_measured as boolean) as t4u_measured,
        cast(FTI_measured as boolean) as fti_measured,
        cast(TBG_measured as boolean) as tbg_measured,

        -- ── Lab values (MinMax-scaled in ingest step, kept as-is here) ──
        cast(TSH as float) as tsh,
        cast(T3  as float) as t3,
        cast(TT4 as float) as tt4,
        cast(T4U as float) as t4u,
        cast(FTI as float) as fti,
        cast(TBG as float) as tbg,

        -- ── Classification target ────────────────────────────────────────
        cast(referral_source as varchar) as referral_source,
        cast(target          as varchar) as target_class,

        -- ── Pipeline metadata ────────────────────────────────────────────
        current_timestamp                      as ingested_at,
        '{{ env_var("SYNTHGEN_RUN_ID", "local") }}' as pipeline_run_id,
        '{{ this.name }}'                      as _dbt_model

    from source

)

select * from staged
