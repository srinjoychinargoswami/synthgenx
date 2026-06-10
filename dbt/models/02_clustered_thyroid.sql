{{
    config(
        materialized = 'table',
        tags          = ['cluster', 'thyroid', 'ml']
    )
}}

/*
    Model  : 02_clustered_thyroid
    Layer  : Intermediate / ML enrichment
    Purpose: Attach K-means cluster assignments (IDs 0–3) computed by
             src/02_cluster.py and map each cluster to a human-readable
             diagnosis class for downstream synthetic generation.

    Cluster → Diagnosis mapping (calibrated from centroid analysis):
        0 → normal        (euthyroid, unremarkable lab values)
        1 → borderline    (mild TSH deviation, possibly subclinical)
        2 → hyperthyroid  (suppressed TSH, elevated T3/T4)
        3 → hypothyroid   (elevated TSH, reduced T3/T4)

    Depends on: ref('01_raw_thyroid')
*/

with

raw as (

    select * from {{ ref('01_raw_thyroid') }}

),

cluster_assignments as (

    -- Cluster labels are written by the Python clustering step to the same
    -- database table.  If running end-to-end, this table is populated before
    -- dbt runs (enforced by Towerfile step ordering).
    select
        record_id,
        cluster_id
    from {{ source('synthgen', 'thyroid_clusters') }}

),

joined as (

    select
        r.*,
        coalesce(c.cluster_id, -1) as cluster_id

    from raw r
    left join cluster_assignments c using (record_id)

),

labeled as (

    select
        *,

        -- ── Cluster → diagnosis class mapping ───────────────────────────
        case cluster_id
            when 0 then 'normal'
            when 1 then 'borderline'
            when 2 then 'hyperthyroid'
            when 3 then 'hypothyroid'
            else        'unassigned'
        end as diagnosis_class,

        -- ── Cluster confidence tier (for downstream filtering) ───────────
        case cluster_id
            when -1 then 'unassigned'
            when 0  then 'high'          -- large, well-separated centroid
            when 1  then 'medium'        -- overlapping boundary region
            when 2  then 'high'
            when 3  then 'high'
        end as cluster_confidence,

        -- ── Derived clinical flags ───────────────────────────────────────
        case
            when tsh < 0.1                        then true
            else false
        end as is_suppressed_tsh,

        case
            when tsh > 0.9                        then true
            else false
        end as is_elevated_tsh,

        case
            when cluster_id in (2, 3)             then true
            else false
        end as is_abnormal_thyroid,

        -- ── Metadata ────────────────────────────────────────────────────
        current_timestamp as clustered_at

    from joined

)

select * from labeled
