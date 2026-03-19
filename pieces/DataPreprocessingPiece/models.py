from pydantic import BaseModel, Field


class InputModel(BaseModel):
    payload: dict = Field(
        default_factory=dict,
        description=(
            "Preprocessing inputs. Expected keys:\n"
            "- `preprocessing_option`: one of `none`, `prediction`, `correction` (default: `none`).\n"
            "- For `prediction`: provide either `dataframe` (preferred) or `data_path`, "
            "plus `save_data_path` (optional), `flag_each_day`, `preprocessor_features`, "
            "and `keep_datetime`.\n"
            "- For `correction`: provide either `dataframe` (preferred) or `data_path`, "
            "plus `save_data_path` (optional), `flag_each_day`, `preprocessor_features` "
            "and optional `test_size` / `load_all_data`."
        ),
    )


class OutputModel(BaseModel):
    message: str = Field(description="Human-readable status message.")
    artifacts: dict = Field(
        default_factory=dict,
        description="Optional outputs (e.g., cleaned dataset URI, dropped rows stats).",
    )
