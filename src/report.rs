use charming::component::{Axis, Grid, Legend, Title};
use charming::{element::AxisType, series::Line, Chart, ImageFormat, ImageRenderer};
use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;

pub struct ReportData {
    n_epochs: u32,
    train_losses: Vec<f64>,
    train_accuracies: Vec<f64>,
    test_losses: Vec<f64>,
    test_accuracies: Vec<f64>,
}

impl ReportData {
    pub fn new(n_epochs: u32) -> ReportData {
        ReportData {
            n_epochs,
            train_losses: Vec::new(),
            train_accuracies: Vec::new(),
            test_losses: Vec::new(),
            test_accuracies: Vec::new(),
        }
    }

    pub fn add(
        &mut self,
        train_loss: f64,
        train_accuracy: f64,
        test_loss: f64,
        test_accuracy: f64,
    ) {
        self.train_losses.push(train_loss);
        self.train_accuracies.push(train_accuracy);
        self.test_losses.push(test_loss);
        self.test_accuracies.push(test_accuracy);
    }

    pub fn is_empty(&self) -> bool {
        self.train_losses.is_empty()
    }

    pub fn print_report(&self, epoch: u32) {
        assert!(!self.is_empty(), "Error: report_data is empty");
        let n_epochs = self.n_epochs;
        let train_loss = self.train_losses.last().unwrap().to_owned();
        let train_accuracy = self.train_accuracies.last().unwrap().to_owned() * 100.;
        let test_loss = self.test_losses.last().unwrap().to_owned();
        let test_accuracy = self.test_accuracies.last().unwrap().to_owned() * 100.;
        if epoch > 1 {
            println!("\r\x1b[6A");
        }
        let report_message = format!(
            "\
        ┌───────────┬────────────────────────────────┬────────────────────────────────┐\n\
        │   Epoch   │            Train               │             Test               │\n\
        ├───────────┼──────────────┬─────────────────┼──────────────┬─────────────────┤\n\
        │ {:4}/{:<4} │ Loss:{:7.4} │ Accuracy:{:5.1}% │ Loss:{:7.4} │ Accuracy:{:5.1}% │\n\
        └───────────┴──────────────┴─────────────────┴──────────────┴─────────────────┘",
            epoch, n_epochs, train_loss, train_accuracy, test_loss, test_accuracy
        );
        println!("{}", report_message);
    }

    pub fn plot_accuracy(&self, output_path: &str) {
        let n_epochs = self.n_epochs;
        let y_train = self.train_accuracies.to_owned();
        let y_test = self.test_accuracies.to_owned();
        let x_data: Vec<String> = (1..=n_epochs).map(|v| v.to_string()).collect();
        let chart = Chart::new()
            .title(
                Title::new()
                    .text("Accuracy over epochs")
                    .text_style(
                        charming::element::TextStyle::new()
                            .font_size(32)
                            .font_style("bold".to_string()),
                    )
                    .left("center"),
            )
            .legend(
                Legend::new()
                    .text_style(charming::element::TextStyle::new().font_size(28))
                    .top("4.5%"),
            )
            .background_color(charming::element::Color::Value("#FFFFFF".to_string()))
            .x_axis(
                Axis::new()
                    .data(x_data)
                    .type_(AxisType::Category)
                    .axis_label(charming::element::AxisLabel::new().font_size(32))
                    .name("Epoch")
                    .name_location(charming::element::NameLocation::Middle)
                    .name_text_style(charming::element::TextStyle::new().font_size(28))
                    .name_gap(40)
                    .split_line(charming::element::SplitLine::new().show(false)),
            )
            .y_axis(
                Axis::new()
                    .scale(true)
                    .type_(AxisType::Value)
                    .axis_label(charming::element::AxisLabel::new().font_size(32))
                    .min(0.0)
                    .max(1.0)
                    .interval(0.1)
                    .name("Accuracy")
                    .name_location(charming::element::NameLocation::Middle)
                    .name_text_style(charming::element::TextStyle::new().font_size(28))
                    .name_gap(60)
                    .split_line(charming::element::SplitLine::new().show(true)),
            )
            .grid(
                Grid::new()
                    .show(true)
                    .left("5%")
                    .top("10%")
                    .right("2.5%")
                    .bottom("7.5%"),
            )
            .series(
                Line::new()
                    .data(y_train)
                    .line_style(charming::element::LineStyle::new().width(10).opacity(0.8))
                    .symbol_size(20)
                    .name("Train"),
            )
            .series(
                Line::new()
                    .name("Validation")
                    .data(y_test)
                    .line_style(charming::element::LineStyle::new().width(10).opacity(0.8))
                    .symbol_size(20),
            );
        let mut renderer = ImageRenderer::new(1920, 1080);
        renderer
            .save_format(ImageFormat::Png, &chart, output_path)
            .expect("Failure when saving plot.");
    }

    pub fn save_training_history(&self, output_path: &str) {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(output_path)
            .expect("Failure when saving training history.");
        for i in 1..self.n_epochs as usize {
            let n_epochs = self.n_epochs;
            let train_loss = self.train_losses[i];
            let train_accuracy = self.train_accuracies[i];
            let test_loss = self.test_losses[i];
            let test_accuracy = self.test_losses[i];
            writeln!(
                file,
                "Epoch {}/{} \
                | Train: Loss {}, Accuracy {} \
                | Test: Loss {}, Accuracy {}",
                i + 1,
                n_epochs,
                train_loss,
                train_accuracy,
                test_loss,
                test_accuracy
            )
            .expect("Failure when saving training history.");
        }
    }

    pub fn save_report(&self, plot: bool, history: bool) {
        create_dir_all("./output").expect("Failure saving report.");
        if plot {
            self.plot_accuracy("output/accuracy.png");
            println!("Accuracy plot saved to: output/accuracy.png");
        }
        if history {
            self.save_training_history("output/training_history.txt");
            println!("Training history saved to: output/training_history.txt");
        }
    }
}
