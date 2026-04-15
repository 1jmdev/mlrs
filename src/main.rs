use std::error::Error;

use mlrs::darray::Array;
use mlrs::preprocessing::MaxAbsScaler;

fn main() -> Result<(), Box<dyn Error>> {
    // Data with different scales
    let x_train =
        Array::from_shape_vec(&[3, 3], vec![1.0, -5.0, 2.0, 3.0, 0.0, 1.0, 0.0, 2.0, -4.0]);

    // New data to transform
    let x_test = Array::from_shape_vec(&[1, 3], vec![6.0, -10.0, 1.0]);

    let mut scaler = MaxAbsScaler::new();

    // Fit on training data only
    scaler.fit(&x_train)?;
    println!("Max abs values: {:?}", scaler.max_abs_);
    println!("Scale factors: {:?}", scaler.scale_);

    // Transform both
    let x_train_scaled = scaler.transform(&x_train)?;
    let x_test_scaled = scaler.transform(&x_test)?;

    println!("Train scaled:\n{:?}", x_train_scaled);
    // [[0.333, -1.0, 0.5], [1.0, 0.0, 0.25], [0.0, 0.4, -1.0]]

    println!("Test scaled: {:?}", x_test_scaled);
    // [[2.0, -2.0, 0.25]]  (uses train's max values: 6/3=2, -10/5=-2, 1/4=0.25)

    Ok(())
}
