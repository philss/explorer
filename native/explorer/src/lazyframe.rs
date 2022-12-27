use crate::{ExDataFrame, ExExpr, ExLazyFrame, ExplorerError};
use polars::prelude::*;
use std::result::Result;

// Loads the IO functions for read/writing CSV, NDJSON, Parquet, etc.
pub mod io;

#[rustler::nif(schedule = "DirtyCpu")]
pub fn lf_collect(data: ExLazyFrame, groups: Vec<String>) -> Result<ExDataFrame, ExplorerError> {
    let lf: LazyFrame = data.resource.0.clone();

    let result = if groups.is_empty() {
        lf.collect()
    } else {
        let cols: Vec<Expr> = groups.iter().map(|g| col(&g)).collect();
        lf.groupby_stable(cols).agg([col("*")]).collect()
    };

    Ok(ExDataFrame::new(result?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn lf_fetch(data: ExLazyFrame, n_rows: usize) -> Result<ExDataFrame, ExplorerError> {
    Ok(ExDataFrame::new(data.resource.0.clone().fetch(n_rows)?))
}

#[rustler::nif]
pub fn lf_describe_plan(data: ExLazyFrame, optimized: bool) -> Result<String, ExplorerError> {
    let lf = &data.resource.0;
    let plan = match optimized {
        true => lf.describe_optimized_plan()?,
        false => lf.describe_plan(),
    };
    Ok(plan)
}

#[rustler::nif]
pub fn lf_head(data: ExLazyFrame, length: u32) -> Result<ExLazyFrame, ExplorerError> {
    let lf = &data.resource.0;
    Ok(ExLazyFrame::new(lf.clone().limit(length)))
}

#[rustler::nif]
pub fn lf_tail(data: ExLazyFrame, length: u32) -> Result<ExLazyFrame, ExplorerError> {
    let lf = &data.resource.0;
    Ok(ExLazyFrame::new(lf.clone().tail(length)))
}

#[rustler::nif]
pub fn lf_names(data: ExLazyFrame) -> Result<Vec<String>, ExplorerError> {
    let lf = &data.resource.0;
    Ok(lf.schema()?.iter_names().cloned().collect())
}

#[rustler::nif]
pub fn lf_dtypes(data: ExLazyFrame) -> Result<Vec<String>, ExplorerError> {
    let lf = &data.resource.0;
    Ok(lf
        .schema()?
        .iter_dtypes()
        .map(|dtype| dtype.to_string())
        .collect())
}

#[rustler::nif]
pub fn lf_select(data: ExLazyFrame, columns: Vec<&str>) -> Result<ExLazyFrame, ExplorerError> {
    let lf = &data.resource.0.clone().select(&[cols(columns)]);
    Ok(ExLazyFrame::new(lf.clone()))
}

#[rustler::nif]
pub fn lf_drop(data: ExLazyFrame, columns: Vec<&str>) -> Result<ExLazyFrame, ExplorerError> {
    let lf = &data.resource.0.clone().select(&[col("*").exclude(columns)]);
    Ok(ExLazyFrame::new(lf.clone()))
}

#[rustler::nif]
pub fn lf_slice(data: ExLazyFrame, offset: i64, length: u32) -> Result<ExLazyFrame, ExplorerError> {
    let lf = data.resource.0.clone();
    Ok(ExLazyFrame::new(lf.slice(offset, length)))
}

#[rustler::nif]
pub fn lf_filter_with(data: ExLazyFrame, ex_expr: ExExpr) -> Result<ExLazyFrame, ExplorerError> {
    let lf: LazyFrame = data.resource.0.clone();
    let exp: Expr = ex_expr.resource.0.clone();

    // We don't consider groups because it's not possible to collect at this point.
    Ok(ExLazyFrame::new(lf.filter(exp)))
}
