use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

fn main() -> Result<(), Box<dyn Error>> {
    // Load CSV file
    let file = File::open("data.csv")?;
    let mut reader = BufReader::new(file);

    // Read entire CSV into ndarray
    let array: Array2<f64> = reader.deserialize_array2_dynamic()?;

    // Separate features and target
    let (features, targets) = array.view().split_at(Axis(1), array.ncols() - 1);
    let targets = targets.column(0).to_owned();

    // Create dataset
    let dataset = Dataset::new(features.to_owned(), targets)
        .with_feature_names(vec![
            "team_strength", "opponent_strength",
            "team_yards", "opponent_yards",
            "team_turnovers", "opponent_turnovers"
        ]);

    // Train-test split
    let (train, valid) = dataset.split_with_ratio(0.8);

    // Train model
    let model = LogisticRegression::default()
        .max_iterations(100)
        .fit(&train)?;

    // Evaluate
    let pred = model.predict(&valid);
    let cm = pred.confusion_matrix(&valid)?;
    println!("Confusion Matrix:\n{}", cm);
    println!("Accuracy: {:.2}%", cm.accuracy() * 100.0);

    // Predict a new game
    let new_game = ndarray::array![
        [0.75, 0.6, 350.0, 280.0, 1.0, 2.0]
    ];
    let new_dataset = Dataset::from(new_game);
    let prediction = model.predict(&new_dataset);
    println!("Prediction for new game: {}", prediction);

    Ok(())
}
