use charming::component::{Axis, Grid, Legend, Title};
use charming::datatype::{CompositeValue, NumericValue};
use charming::element::{
    AxisLabel, AxisType, Color, Label, LineStyle, MarkLine, MarkLineData, MarkLineVariant,
    NameLocation, SplitLine, Symbol, TextStyle,
};
use charming::series::Line;
use charming::{Chart, ImageFormat, ImageRenderer};

pub fn plot_cdf(sorted_errors: Vec<f64>, cdf: Vec<f64>, output_path: &str) {
    let line_data: Vec<CompositeValue> = sorted_errors
        .clone()
        .into_iter()
        .zip(cdf.clone().into_iter())
        .map(|(x, y)| CompositeValue::from(vec![NumericValue::from(x), NumericValue::from(y)]))
        .collect();
    let mut cdf_90_probability = (0., 0.);
    for (i, &prob) in cdf.iter().enumerate() {
        if prob >= 0.9 {
            cdf_90_probability.0 = cdf[i];
            cdf_90_probability.1 = sorted_errors[i];
            break;
        }
    }
    let chart = Chart::new()
        .title(
            Title::new()
                .text("CDF of Distance Errors (RMSE)")
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
                .type_(AxisType::Value)
                .interval(1)
                .axis_label(AxisLabel::new().font_size(32))
                .name("Distance Error")
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
                .name("Cumulative Probability")
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
                .name("CDF")
                .data(line_data)
                .line_style(LineStyle::new().width(5).opacity(1.0))
                .show_symbol(false)
                .smooth(false)
                .mark_line(
                    MarkLine::new()
                        .line_style(LineStyle::new().width(2.5).color("#2D2D2D"))
                        .symbol(vec![Symbol::None, Symbol::None])
                        .label(Label::new().show(false))
                        .data(vec![
                            MarkLineVariant::Simple(
                                MarkLineData::new().x_axis(cdf_90_probability.1),
                            ),
                            MarkLineVariant::Simple(
                                MarkLineData::new().y_axis(cdf_90_probability.0),
                            ),
                        ]),
                ),
        );

    let mut renderer = ImageRenderer::new(1920, 1080);
    renderer
        .save_format(ImageFormat::Png, &chart, output_path)
        .expect("Failure when saving plot.");
}
