from domino.base_piece import BasePiece



from .models import InputModel, OutputModel





class ElectricityPricePredictionModelTrainPiece(BasePiece):

    def piece_function(self, input_data: InputModel):

        import csv

        import datetime as dt

        import json

        import os

        import tempfile

        from collections import defaultdict



        import joblib

        import numpy as np

        import pandas as pd

        import requests

        from zoneinfo import ZoneInfo



        self.logger.info("Running ElectricityPricePredictionModelTrainPiece.")

        payload = input_data.payload or {}

        _brat = ZoneInfo("Europe/Bratislava")



        def _parse_dt(value):

            if value is None:

                return None

            if isinstance(value, dt.datetime):

                return value

            text = str(value).strip()

            if not text:

                return None

            text = text.replace("Z", "+00:00")

            try:

                return dt.datetime.fromisoformat(text)

            except ValueError:

                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):

                    try:

                        return dt.datetime.strptime(text, fmt)

                    except ValueError:

                        continue

            return None



        def _normalize_slot_key(ts: dt.datetime) -> str:

            minute = (ts.minute // 15) * 15

            return f"{ts.hour:02d}:{minute:02d}"



        def _to_bratislava_naive(ts: dt.datetime) -> dt.datetime:

            if ts.tzinfo is None:

                return ts.replace(tzinfo=_brat).astimezone(_brat).replace(tzinfo=None)

            return ts.astimezone(_brat).replace(tzinfo=None)



        def _brat_date_and_slot(ts: dt.datetime) -> tuple[str, str]:

            local = _to_bratislava_naive(ts)

            return local.date().isoformat(), _normalize_slot_key(local)



        def _read_input_rows(raw_payload: dict) -> list[dict]:

            data_path = raw_payload.get("data_path") or raw_payload.get("csv_path")

            tabular_data = raw_payload.get("tabular_data") or raw_payload.get("dataframe")

            if data_path:

                with open(data_path, "r", encoding="utf-8") as f:

                    return list(csv.DictReader(f))

            if isinstance(tabular_data, list):

                return tabular_data

            if isinstance(tabular_data, dict):

                keys = list(tabular_data.keys())

                n = len(tabular_data[keys[0]]) if keys else 0

                return [{k: tabular_data[k][i] for k in keys} for i in range(n)]

            raise ValueError(

                "Provide either `payload['data_path']`/`payload['csv_path']` or `payload['tabular_data']`."

            )



        def _extract_row_datetime(row: dict, datetime_keys: tuple[str, ...]) -> dt.datetime | None:

            for key in datetime_keys:

                if key in row:

                    parsed = _parse_dt(row.get(key))

                    if parsed is not None:

                        return parsed

            for key in ("datetime", "datetime_15m", "timestamp", "ts"):

                parsed = _parse_dt(row.get(key))

                if parsed is not None:

                    return parsed

            return None



        def _iter_okte_rows(start_date: dt.date, end_date: dt.date) -> list[dict]:

            endpoint = str(

                (payload.get("okte") or {}).get("endpoint")

                or "https://isot.okte.sk/api/v1/dam/results"

            )

            candidate_params = [

                {

                    "deliverydayfrom": start_date.isoformat(),

                    "deliverydayto": end_date.isoformat(),

                },

                {

                    "delivery_from": start_date.isoformat(),

                    "delivery_to": end_date.isoformat(),

                },

                {

                    "deliveryFrom": start_date.isoformat(),

                    "deliveryTo": end_date.isoformat(),

                },

                {"date_from": start_date.isoformat(), "date_to": end_date.isoformat()},

            ]

            for params in candidate_params:

                try:

                    response = requests.get(endpoint, params=params, timeout=20)

                    if response.status_code != 200:

                        continue

                    body = response.json()

                except Exception:

                    continue

                if isinstance(body, list):

                    return body

                if isinstance(body, dict):

                    for key in ("results", "data", "items"):

                        value = body.get(key)

                        if isinstance(value, list):

                            return value

            return []



        def _extract_price_map(okte_rows: list[dict]) -> dict[tuple[str, str], float]:

            price_map: dict[tuple[str, str], float] = {}

            for row in okte_rows:

                if not isinstance(row, dict):

                    continue

                date_text = (

                    row.get("deliveryDay")

                    or row.get("delivery_date")

                    or row.get("deliveryDate")

                    or row.get("date")

                    or row.get("trading_date")

                )

                date_obj = None

                if date_text:

                    try:

                        date_obj = dt.date.fromisoformat(str(date_text)[:10])

                    except ValueError:

                        date_obj = None

                dt_val = _parse_dt(

                    row.get("datetime")

                    or row.get("delivery_datetime")

                    or row.get("deliveryStart")

                    or row.get("delivery_start")

                )



                if dt_val is not None and (

                    row.get("deliveryStart") or row.get("delivery_start")

                ):

                    local = _to_bratislava_naive(dt_val)

                    map_date = local.date().isoformat()

                    slot_key = _normalize_slot_key(local)

                    price = (

                        row.get("price_eur_mwh")

                        or row.get("priceEurMwh")

                        or row.get("price")

                        or row.get("dam_price")

                    )

                    try:

                        price_map[(map_date, slot_key)] = float(price)

                    except (TypeError, ValueError):

                        pass

                    continue



                if date_obj is None and dt_val is not None:

                    date_obj = _to_bratislava_naive(dt_val).date()

                if date_obj is None:

                    continue



                hour_raw = row.get("hour") or row.get("delivery_hour") or row.get("slot")

                quarter_raw = row.get("quarter") or row.get("quarter_hour")

                if dt_val is not None:

                    slot_key = _normalize_slot_key(_to_bratislava_naive(dt_val))

                elif hour_raw is not None:

                    try:

                        hour_int = int(hour_raw)

                        if 1 <= hour_int <= 24:

                            hour_int = (hour_int - 1) % 24

                        quarter_int = int(quarter_raw) if quarter_raw is not None else 0

                        minute = max(0, min(3, quarter_int)) * 15

                        slot_key = f"{hour_int:02d}:{minute:02d}"

                    except (TypeError, ValueError):

                        continue

                else:

                    continue



                price = (

                    row.get("price_eur_mwh")

                    or row.get("priceEurMwh")

                    or row.get("price")

                    or row.get("dam_price")

                )

                try:

                    price_value = float(price)

                except (TypeError, ValueError):

                    continue

                price_map[(date_obj.isoformat(), slot_key)] = price_value

            return price_map



        def _enrich_rows_with_okte(

            input_rows: list[dict],

            row_datetimes: list[dt.datetime],

            dt_col: str,

        ) -> tuple[list[dict], bool]:

            target_start = min(row_datetimes).date()

            target_end = max(row_datetimes).date()



            direct_okte_rows = _iter_okte_rows(target_start, target_end)

            direct_price_map = _extract_price_map(direct_okte_rows)



            fallback_used = False

            enriched_rows: list[dict] = []



            fallback_end = dt.datetime.utcnow().date()

            fallback_start = fallback_end - dt.timedelta(days=28)

            fallback_okte_rows = _iter_okte_rows(fallback_start, fallback_end)

            fallback_price_map = _extract_price_map(fallback_okte_rows)



            slot_prices: dict[str, list[float]] = defaultdict(list)

            for (date_key, slot_key), value in fallback_price_map.items():

                if date_key and slot_key:

                    slot_prices[slot_key].append(value)

            slot_avg = {

                slot: (sum(values) / len(values))

                for slot, values in slot_prices.items()

                if values

            }



            global_avg = None

            all_prices = list(direct_price_map.values()) + list(fallback_price_map.values())

            if all_prices:

                global_avg = sum(all_prices) / len(all_prices)



            for row, row_ts in zip(input_rows, row_datetimes):

                date_key, slot_key = _brat_date_and_slot(row_ts)

                price = direct_price_map.get((date_key, slot_key))

                if price is None:

                    fallback_used = True

                    price = slot_avg.get(slot_key, global_avg)

                if price is None:

                    raise ValueError(

                        "Could not resolve electricity price from OKTE and fallback averages are unavailable."

                    )

                enriched = dict(row)

                enriched["datetime"] = row_ts.isoformat(sep=" ")

                enriched["price_eur_mwh"] = float(price)

                enriched["price_source"] = (

                    "okte_direct" if (date_key, slot_key) in direct_price_map else "fallback_28d_avg"

                )

                enriched_rows.append(enriched)



            return enriched_rows, fallback_used



        setup = payload.get("model_setup") or {}

        feature_columns = setup.get("feature_columns")

        if not feature_columns or not isinstance(feature_columns, list):

            raise ValueError(

                "`payload['model_setup']['feature_columns']` must be a non-empty list of column names."

            )

        feature_columns = [str(c) for c in feature_columns]



        target_column = str(setup.get("target_column", "price_eur_mwh"))

        datetime_column = str(setup.get("datetime_column", "datetime"))

        target_source = str(setup.get("target_source", "column")).lower().strip()

        if target_source not in ("column", "okte"):

            raise ValueError(

                "model_setup.target_source must be 'column' (labels already in data) "

                "or 'okte' (fetch DAM prices for each row's delivery slot)."

            )



        input_rows = _read_input_rows(payload)

        if not input_rows:

            raise ValueError("No rows were provided in input payload.")



        dt_keys = (datetime_column,)

        datetimes = [_extract_row_datetime(row, dt_keys) for row in input_rows]

        if any(ts is None for ts in datetimes):

            raise ValueError(

                f"Each row must contain parseable datetime in `{datetime_column}` "

                "or fallback keys: datetime, datetime_15m, timestamp, ts."

            )

        row_datetimes = [ts for ts in datetimes if ts is not None]



        fallback_used = False

        if target_source == "okte":

            training_rows, fallback_used = _enrich_rows_with_okte(
                input_rows, row_datetimes, datetime_column
            )

        else:

            training_rows = []

            for row, row_ts in zip(input_rows, row_datetimes):

                r = dict(row)

                r[datetime_column] = row_ts.isoformat(sep=" ")

                training_rows.append(r)



        df = pd.DataFrame(training_rows)



        missing_feat = [c for c in feature_columns if c not in df.columns]

        if missing_feat:

            raise ValueError(f"Missing feature columns in data: {missing_feat}")



        if target_column not in df.columns:

            raise ValueError(

                f"Missing target column `{target_column}` in training data. "

                "Use target_source='okte' to fill DAM prices into price_eur_mwh."

            )



        for col in feature_columns + [target_column]:

            df[col] = pd.to_numeric(df[col], errors="coerce")



        df = df.dropna(subset=feature_columns + [target_column])

        if len(df) < 2:

            raise ValueError("Need at least 2 valid rows after dropping NaNs in features and target.")



        X = df[feature_columns].to_numpy(dtype=float)

        y = df[target_column].to_numpy(dtype=float)



        from xgboost import XGBRegressor



        xgb_params = dict(payload.get("xgb_params") or {})

        default_params = {

            "n_estimators": 100,

            "max_depth": 6,

            "learning_rate": 0.1,

            "subsample": 0.9,

            "colsample_bytree": 0.9,

            "random_state": 42,

            "n_jobs": -1,

        }

        for k, v in default_params.items():

            xgb_params.setdefault(k, v)



        model = XGBRegressor(**xgb_params)

        model.fit(X, y)

        y_pred = model.predict(X)

        train_metrics = {

            "rmse": float(np.sqrt(np.mean((y - y_pred) ** 2))),

            "mae": float(np.mean(np.abs(y - y_pred))),

        }



        output_dir = payload.get("output_dir")

        if not output_dir:

            output_dir = str(getattr(self, "results_path", None) or tempfile.gettempdir())

        os.makedirs(output_dir, exist_ok=True)



        model_filename = str(payload.get("model_filename", "electricity_price_xgb.joblib"))

        if not model_filename.endswith((".joblib", ".pkl")):

            model_filename = f"{model_filename}.joblib"

        model_path = os.path.join(output_dir, model_filename)



        joblib.dump(model, model_path)



        metadata = {

            "feature_columns": feature_columns,

            "feature_columns_used": feature_columns,

            "target_column": target_column,

            "datetime_column": datetime_column,

            "target_source": target_source,

            "model_class": "xgboost.sklearn.XGBRegressor",

            "train_rows": int(len(df)),

        }

        metadata_path = os.path.join(output_dir, "preprocessing_metadata.json")

        with open(metadata_path, "w", encoding="utf-8") as f:

            json.dump(metadata, f, indent=2)



        artifacts: dict = {

            "model_path": model_path,

            "preprocessing_metadata_path": metadata_path,

            "train_metrics": train_metrics,

            "train_rows": int(len(df)),

            "fallback_used": fallback_used,

        }



        if bool(payload.get("save_enriched_csv")) and target_source == "okte":

            output_path = os.path.join(output_dir, "electricity_price_enriched.csv")

            fieldnames = sorted({key for row in training_rows for key in row.keys()})

            with open(output_path, "w", newline="", encoding="utf-8") as f:

                writer = csv.DictWriter(f, fieldnames=fieldnames)

                writer.writeheader()

                writer.writerows(training_rows)

            artifacts["enriched_csv_path"] = output_path



        return OutputModel(

            message="Electricity price XGBoost regressor trained and saved.",

            artifacts=artifacts,

        )


