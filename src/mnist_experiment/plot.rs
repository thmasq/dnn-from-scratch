use charming::component::{Axis, Grid, Legend, Title};
use charming::element::{
    AxisLabel, AxisType, Color, LineStyle, NameLocation, SplitLine, TextStyle,
};
use charming::series::Line;
use charming::{Chart, ImageFormat, ImageRenderer};

pub fn plot_error(n_epochs: u32, y_train: Vec<f64>, y_test: Vec<f64>, output_path: &str) {
    let x_data: Vec<String> = (1..=n_epochs).map(|v| v.to_string()).collect();
    let chart = Chart::new()
        .title(
            Title::new()
                .text("Accuracy over epochs")
                .text_style(
                    TextStyle::new()
                        .font_size(32)
                        .font_style("bold".to_string()),
                )
                .left("center"),
        )
        .legend(
            Legend::new()
                .text_style(TextStyle::new().font_size(28))
                .top("4.5%"),
        )
        .background_color(Color::Value("#FFFFFF".to_string()))
        .x_axis(
            Axis::new()
                .data(x_data)
                .type_(AxisType::Category)
                .axis_label(AxisLabel::new().font_size(32))
                .name("Epoch")
                .name_location(NameLocation::Middle)
                .name_text_style(TextStyle::new().font_size(28))
                .name_gap(40)
                .split_line(SplitLine::new().show(false)),
        )
        .y_axis(
            Axis::new()
                .scale(true)
                .type_(AxisType::Value)
                .axis_label(AxisLabel::new().font_size(32))
                .min(0.0)
                .max(1.0)
                .interval(0.1)
                .name("Accuracy")
                .name_location(NameLocation::Middle)
                .name_text_style(TextStyle::new().font_size(28))
                .name_gap(60)
                .split_line(SplitLine::new().show(true)),
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
                .line_style(LineStyle::new().width(10).opacity(0.8))
                .symbol_size(20)
                .name("Train")
                .smooth(false),
        )
        .series(
            Line::new()
                .name("Validation")
                .data(y_test)
                .line_style(LineStyle::new().width(10).opacity(0.8))
                .symbol_size(20)
                .smooth(false),
        );
    let mut renderer = ImageRenderer::new(1920, 1080);
    renderer
        .save_format(ImageFormat::Png, &chart, output_path)
        .expect("Failure when saving plot.");
}
