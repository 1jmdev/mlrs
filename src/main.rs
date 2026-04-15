use mlrs::{darray::np, linear_model::LinearRegression};
use std::{error::Error, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let x = np::array(&[1.0, 2.0, 3.0, 4.0, 5.0]).reshape(&[-1, 1]);
    let y = np::array(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut model = LinearRegression::new()
        .epochs(1000)
        .learning_rate(0.01)
        .fit_intercept(true);

    let start = Instant::now();
    model.fit(&x, &y)?;
    println!("Train time: {:?}", start.elapsed());

    let coefficient = model.coef()?.get(&[0]);
    println!("Learned multiplier: {coefficient:?}");

    let predict_input = np::array(&[7.0]).reshape(&[-1, 1]);
    let prediction = model.predict(&predict_input)?;
    println!("Prediction for 7: {:?}", prediction.get(&[0]));

    Ok(())
}
