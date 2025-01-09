use nd::{s, Array2};
use polars::prelude::*;

pub fn load_rssi_dataset(
    path_to_csv: &str,
    test_proportion: f64,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    assert!(
        0. < test_proportion && test_proportion < 1.,
        "The test proportion should be in the (0, 1) range."
    );
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path_to_csv.into()))
        .unwrap()
        .finish()
        .unwrap()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let df_nrows = df.nrows();
    let mut y_matrix = Array2::zeros((df_nrows, 2));
    let mut x_matrix = Array2::zeros((df_nrows, 13));
    for i in 0..df_nrows {
        for j in 1..=2 {
            y_matrix[[i, j - 1]] = df[[i, j]];
        }
        for j in 3..=15 {
            x_matrix[[i, j - 3]] = df[[i, j]];
        }
    }
    let num_test = (df_nrows as f64 * test_proportion).round() as usize;
    let num_train = df_nrows - num_test;
    let train_slice = s![0..num_train, ..];
    let test_slice = s![num_train..df_nrows, ..];
    let x_train = x_matrix.slice(train_slice).into_owned();
    let y_train = y_matrix.slice(train_slice).into_owned();
    let x_test = x_matrix.slice(test_slice).into_owned();
    let y_test = y_matrix.slice(test_slice).into_owned();
    (x_train, y_train, x_test, y_test)
}
