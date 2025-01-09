use nd::{Array2, Array3, ArrayBase, ArrayD, Axis, Dimension, Ix3};
use npy::ReadNpyExt;
use std::fs::File;
use std::path::{Path, PathBuf};

fn to_array3<T, D>(array: ArrayBase<T, D>) -> ArrayBase<T, Ix3>
where
    T: ndarray::Data,
    D: Dimension,
{
    match array.ndim() {
        1 => {
            // If it's 1D, add two axes to make it a 3D array
            array
                .insert_axis(Axis(1))
                .insert_axis(Axis(2))
                .into_dimensionality::<Ix3>()
                .unwrap()
        }
        2 => {
            // If it's 2D, add one axis to make it a 3D array
            array
                .insert_axis(Axis(2))
                .into_dimensionality::<Ix3>()
                .unwrap()
        }
        3 => {
            // If it's already 3D, return as-is
            array.into_dimensionality::<Ix3>().unwrap()
        }
        _ => panic!("Unsupported dimensionality: {}", array.ndim()),
    }
}

fn read_mnist_npy(path_to_npy: PathBuf) -> Array3<f64> {
    let reader = File::open(path_to_npy).expect("Failure when reading npy file.");
    let array = ArrayD::<u8>::read_npy(reader).expect("Failure when parsing npy file.");
    let array = array.mapv(|v| v as f64);
    let array = to_array3(array);
    array
}

fn process_images(image_batch: &Array3<f64>) -> Array2<f64> {
    let nrows = image_batch.shape()[0];
    let ncols = image_batch.shape()[1] * image_batch.shape()[2];
    let reshaped_images = image_batch
        .clone()
        .into_shape_clone((nrows, ncols))
        .unwrap();
    let output = reshaped_images.clone();
    output
}

fn normalize_images(processed_images: &Array2<f64>) -> Array2<f64> {
    let normalized_images = processed_images.mapv(|v| v / 255.);
    normalized_images
}

fn to_categorical(x_train: &Array2<f64>) -> Array2<f64> {
    let num_classes = x_train.fold(0.0, |acc: f64, &x| acc.max(x)) as usize + 1;
    let mut result = Array2::zeros((x_train.nrows(), num_classes));
    for (i, row) in x_train.outer_iter().enumerate() {
        if let Some(&label) = row.iter().next() {
            if label >= 0.0 && label < num_classes as f64 {
                result[[i, label as usize]] = 1.0;
            }
        }
    }
    result
}

pub fn load_mnist_dataset(
    path_to_folder: &str,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let files = ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"];
    let path_to_folder = Path::new(path_to_folder);
    let mut arrays = [None, None, None, None];
    for (i, &file) in files.iter().enumerate() {
        let file = Path::new(file);
        let filepath = path_to_folder.join(file);
        let array = read_mnist_npy(filepath);
        let processed_array = process_images(&array);
        if (i as i32) % 2 == 0 {
            let normalized_array = normalize_images(&processed_array);
            arrays[i] = Some(normalized_array);
        } else {
            let categorized_array = to_categorical(&processed_array);
            arrays[i] = Some(categorized_array);
        }
    }
    let (x_train, y_train, x_test, y_test) = (
        arrays[0].take().unwrap(),
        arrays[1].take().unwrap(),
        arrays[2].take().unwrap(),
        arrays[3].take().unwrap(),
    );
    (x_train, y_train, x_test, y_test)
}
