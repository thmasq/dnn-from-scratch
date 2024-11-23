use na::DMatrix;

pub fn load_dataset() -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
    let (nrows, ncols) = (10, 10);
    let x_train: DMatrix<f64> = DMatrix::zeros(nrows, ncols);
    let y_train: DMatrix<f64> = DMatrix::zeros(nrows, ncols);
    let x_test: DMatrix<f64> = DMatrix::zeros(nrows, ncols);
    let y_test: DMatrix<f64> = DMatrix::zeros(nrows, ncols);
    (x_train, y_train, x_test, y_test)
}
