use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;

enum ErrorMetric {
    Accuracy,
    Error,
}

pub struct ReportData {
    n_epochs: u32,
    train_losses: Vec<f64>,
    train_errors: Vec<f64>,
    test_losses: Vec<f64>,
    test_errors: Vec<f64>,
    metric: ErrorMetric,
}

impl ReportData {
    pub fn new(n_epochs: u32, error_metric: &str) -> ReportData {
        let metric = match error_metric {
            "accuracy" => ErrorMetric::Accuracy,
            "error" => ErrorMetric::Error,
            _ => panic!("Unintended error metric passed."),
        };
        ReportData {
            n_epochs,
            train_losses: Vec::new(),
            train_errors: Vec::new(),
            test_losses: Vec::new(),
            test_errors: Vec::new(),
            metric,
        }
    }

    pub fn add(&mut self, train_loss: f64, train_error: f64, test_loss: f64, test_error: f64) {
        self.train_losses.push(train_loss);
        self.train_errors.push(train_error);
        self.test_losses.push(test_loss);
        self.test_errors.push(test_error);
    }

    pub fn get_errors(&self) -> (Vec<f64>, Vec<f64>) {
        (self.train_errors.clone(), self.test_errors.clone())
    }

    pub fn get_losses(&self) -> (Vec<f64>, Vec<f64>) {
        (self.train_losses.clone(), self.test_losses.clone())
    }

    pub fn is_empty(&self) -> bool {
        self.train_losses.is_empty()
    }

    pub fn print_report(&self, epoch: u32) {
        assert!(!self.is_empty(), "Error: report_data is empty");
        let n_epochs = self.n_epochs;
        let train_loss = self.train_losses.last().unwrap().to_owned();
        let train_error = self.train_errors.last().unwrap().to_owned();
        let test_loss = self.test_losses.last().unwrap().to_owned();
        let test_error = self.test_errors.last().unwrap().to_owned();
        if epoch > 1 {
            println!("\r\x1b[6A");
        }
        let report_message = match self.metric {
            ErrorMetric::Accuracy => {
                format!(
                    "\
                ┌───────────┬────────────────────────────────┬────────────────────────────────┐  \n\
                │   Epoch   │            Train               │             Test               │  \n\
                ├───────────┼──────────────┬─────────────────┼──────────────┬─────────────────┤  \n\
                │ {:4}/{:<4} │ Loss:{:7.4} │ Accuracy:{:5.1}% │ Loss:{:7.4} │ Accuracy:{:5.1}% │  \n\
                └───────────┴──────────────┴─────────────────┴──────────────┴─────────────────┘  ",
                    epoch,
                    n_epochs,
                    train_loss,
                    train_error * 100.,
                    test_loss,
                    test_error * 100.
                )
            }
            ErrorMetric::Error => {
                format!(
                    "\
                ┌───────────┬────────────────────────────────┬────────────────────────────────┐  \n\
                │   Epoch   │            Train               │             Test               │  \n\
                ├───────────┼──────────────┬─────────────────┼──────────────┬─────────────────┤  \n\
                │ {:4}/{:<4} │ Loss:{:7.2} │ Error:{:9.2} │ Loss:{:7.2} │ Error:{:9.2} │  \n\
                └───────────┴──────────────┴─────────────────┴──────────────┴─────────────────┘  ",
                    epoch, n_epochs, train_loss, train_error, test_loss, test_error
                )
            }
        };
        println!("{}", report_message);
    }

    pub fn save_training_history(&self, output_path: &str) {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(output_path)
            .expect("Failure when saving training history.");
        for i in 0..self.n_epochs as usize {
            let n_epochs = self.n_epochs;
            let train_loss = self.train_losses[i];
            let train_error = self.train_errors[i];
            let test_loss = self.test_losses[i];
            let test_error = self.test_errors[i];
            let metric = match self.metric {
                ErrorMetric::Accuracy => "Accuracy",
                ErrorMetric::Error => "Error",
            };
            writeln!(
                file,
                "Epoch {}/{} \
                | Train: Loss {:.8}, {} {:.8} \
                | Test: Loss {:.8}, {} {:.8}",
                i + 1,
                n_epochs,
                train_loss,
                metric,
                train_error,
                test_loss,
                metric,
                test_error
            )
            .expect("Failure when saving training history.");
        }
    }

    pub fn save_report(&self, output_file: &str) {
        create_dir_all("./output").expect("Failure saving report.");
        let output_file: &str = &format!("output/{}", output_file);
        self.save_training_history(output_file);
        println!("Training history saved to: {}", output_file);
    }
}
