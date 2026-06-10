{{
    config(
        materialized = 'table',
        tags          = ['synthetic', 'thyroid', 'feature-engineering']
    )
}}

/*
    Model  : 03_synthetic_ready
    Layer  : Feature / Output
    Purpose: Final feature table consumed by the synthetic data generator
             (src/03_synthetic_gen.py).  Normalises continuous features to
             [0, 1] so the generative model sees a consistent input space,
             and selects only the columns needed for synthesis.

    Normalisation approach
    ──────────────────────
    age  : divide by 100 (reasonable physiological upper bound)
    TSH  : already MinMax-scaled by ingest step; kept as-is
           (stored in [0,1] after 01_ingest.py MinMaxScaler pass)
    T3, TT4, T4U, FTI: same — already [0,1] from ingest step

    Note: if raw (un-scaled) values are loaded, replace the pass-through
    expressions with proper window-based min–max expressions shown below.

    Depends on: ref('02_clustered_thyroid')
*/

with

clustered as (

    select * from {{ ref('02_clustered_thyroid') }}

),

/*
    Optional: compute per-column min/max for in-SQL normalisation when
    values are NOT pre-scaled by the ingest step.

    stats as (
        select
            min(age)  as age_min,  max(age)  as age_max,
            min(tsh)  as tsh_min,  max(tsh)  as tsh_max,
            min(t3)   as t3_min,   max(t3)   as t3_max,
            min(tt4)  as tt4_min,  max(tt4)  as tt4_max,
            min(t4u)  as t4u_min,  max(t4u)  as t4u_max,
            min(fti)  as fti_min,  max(fti)  as fti_max
        from clustered
    ),
*/

normalized as (

    select
        -- ── Primary key ─────────────────────────────────────────────────
        record_id                                       as id,

        -- ── Target label ────────────────────────────────────────────────
        diagnosis_class,
        cluster_id,
        cluster_confidence,

        -- ── Raw feature columns (ingest-scaled [0,1]) ───────────────────
        age                                             as age,
        tsh                                             as tsh,
        t3                                              as t3,
        tt4                                             as tt4,
        t4u                                             as t4u,
        fti                                             as fti,

        -- ── Normalised feature columns ───────────────────────────────────
        --    age:  scale to physiological [0,1] via /100 cap
        round(
            least(greatest(age / 100.0, 0.0), 1.0),
            6
        )                                               as age_normalized,

        --    TSH: already [0,1]; clip for any edge cases
        round(
            least(greatest(tsh, 0.0), 1.0),
            6
        )                                               as tsh_normalized,

        --    T3: already [0,1]; clip
        round(
            least(greatest(t3, 0.0), 1.0),
            6
        )                                               as t3_normalized,

        --    TT4: already [0,1]; clip
        round(
            least(greatest(tt4, 0.0), 1.0),
            6
        )                                               as tt4_normalized,

        --    T4U: already [0,1]; clip
        round(
            least(greatest(t4u, 0.0), 1.0),
            6
        )                                               as t4u_normalized,

        --    FTI: already [0,1]; clip
        round(
            least(greatest(fti, 0.0), 1.0),
            6
        )                                               as fti_normalized,

        -- ── Clinical flags (useful conditioning signals for GAN/VAE) ────
        is_suppressed_tsh,
        is_elevated_tsh,
        is_abnormal_thyroid,

        -- ── Demographic passthrough ──────────────────────────────────────
        sex,
        on_thyroxine,
        on_antithyroid_medication,
        thyroid_surgery,

        -- ── Pipeline metadata ────────────────────────────────────────────
        ingested_at,
        clustered_at,
        current_timestamp                               as feature_ready_at,
        pipeline_run_id

    from clustered

    -- Exclude unassigned records from the synthesis-ready table
    where cluster_id != -1

),

/*
    Quality gate: remove records where any key feature is NULL.
    These should be zero after the ingest imputation step, but this
    acts as a defensive check.
*/
quality_checked as (

    select *
    from normalized
    where
        id                is not null
        and diagnosis_class is not null
        and age_normalized  is not null
        and tsh_normalized  is not null

)

select * from quality_checked
