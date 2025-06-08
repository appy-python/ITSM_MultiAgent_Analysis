import os
from datetime import timedelta
from feast import Entity, Feature, FeatureView, Field, ValueType
from feast.types import Float32, Int64, Bool
from feast.infra.offline_stores.file_source import FileSource


# Add value_type to the entity
itsm_entity = Entity(
    name="incident_id",
    join_keys=["Incident_ID"],
    value_type=ValueType.STRING  # Add this line to specify the value type
)


# Be more explicit about the timestamp column
itsm_data_source = FileSource(
    path="data/itsm_cleaned.parquet",
    event_timestamp_column="Open_Time",
    timestamp_field="Open_Time",  # Add this line to be explicit about timestamp field
    #created_timestamp_column="Open_Time"  # Optional: Add if you have a creation timestamp
)


# Now create the feature view with the entity object
itsm_feature_view = FeatureView(
    name="itsm_features",
    entities=[itsm_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="Handle_Time_hrs", dtype=Float32),
        Field(name="CI_Name_enc", dtype=Int64),
        Field(name="CI_Cat_enc", dtype=Int64),
        Field(name="CI_Subcat_enc", dtype=Int64),
        Field(name="Closure_Code_enc", dtype=Int64),
        Field(name="Priority_enc", dtype=Int64),
        Field(name="Impact_enc", dtype=Int64),
        Field(name="Urgency_enc", dtype=Int64),
        Field(name="Resolution_SLA_Breach", dtype=Bool),
        Field(name="Has_KB", dtype=Int64),  # Optional
    ],
    online=True,
    source=itsm_data_source,
)

