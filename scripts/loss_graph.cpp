#include <iostream>
#include <vector>
#include <string>
#include <fstream>


struct Serie {
	std::vector<std::pair<float, float>> data;
	std::string name;
	std::string color;
	std::string marker;
};

Serie read_csv(std::string& path) {
	Serie serie;

	std::ifstream f(path);
	if (f.is_open()) {
		std::string line;
		getline(f, line); // Remove header
		while (getline(f, line)) {
			std::string x_str = "";
			std::string y_str = "";
			bool is_x = true;
			for (int i = 0; i < line.size(); i++) {
				const char ch = line[i];
				if (ch == ',') {
					is_x = false;
					continue;
				}
				if (is_x) x_str.push_back(ch);
				else y_str.push_back(ch);
			}
			
			serie.data.push_back({std::stof(x_str), std::stof(y_str)});
		}
		f.close();
	} else {
		throw std::runtime_error("Cannot open file: " + path);
	}

	return serie;
}

std::vector<float> get_bounds(std::vector<Serie> &series) {
	float x_min = 1e9, x_max = -1e9, y_min = 1e9, y_max = -1e9;
	for (const auto &serie : series) {
		for (const auto &p : serie.data) {
			x_min = std::min(x_min, p.first);
			x_max = std::max(x_max, p.first);
			y_min = std::min(y_min, p.second);
			y_max = std::max(y_max, p.second);
		}
	}

	return std::vector<float>{x_min, x_max, y_min, y_max};
}

void create_latex(std::vector<Serie> &series, const std::string &path) {
	std::ofstream f(path);
	if (f.is_open()) {

		std::vector<float> bounds = get_bounds(series);
		float x_min = bounds[0];
		float x_max = bounds[1];
		float y_min = bounds[2];
		float y_max = bounds[3];

		f << "\\documentclass[border=5pt]{standalone}" << std::endl;
		f << "\\usepackage{pgfplots}" << std::endl;
		f << "\\pgfplotsset{compat=1.18}" << std::endl;
		f << std::endl;
		f << "\\begin{document}" << std::endl;
		f << "\\begin{tikzpicture}" << std::endl;
		f << "\\begin{axis}[" << std::endl;
		f << "    width=14cm," << std::endl;
		f << "    height=9cm," << std::endl;
		f << "    xlabel={Iteration}," << std::endl;
		f << "    ylabel={Loss}," << std::endl;
		f << "    legend pos=north east," << std::endl;
		f << "    grid=major," << std::endl;
		f << "    grid style={dashed,gray!30}," << std::endl;
		f << "    xmin=" << x_min << ", xmax=" << x_max << "," << std::endl;
		f << "    ymin=" << y_min << ", ymax=" << y_max << "," << std::endl;
		f << "]" << std::endl;
		f << std::endl;

		for (const auto &serie : series) {
			f << "\\addplot[" << serie.color << ", thick] coordinates {" << std::endl;
			for (const auto& p : serie.data) {
				f << "(" << p.first << "," << p.second << ") ";
			}
			f << std::endl << "};" << std::endl;
			f << "\\addlegendentry{" << serie.name << "}" << std::endl << std::endl;
		}

		f << "\\end{axis}" << std::endl;
		f << "\\end{tikzpicture}" << std::endl;
		f << "\\end{document}" << std::endl;

		f.close();
	} else {
		throw std::runtime_error("Cannot open file: " + path);
	}
}

int main() {
	std::vector<std::string> paths = {"data/muon_loss.csv", "data/adamw_loss.csv", "data/adam_loss.csv"};
	std::vector<std::string> names = {"muon", "adamw", "adam"};
	std::vector<std::string> colors = {"red", "blue", "green"};
	std::vector<std::string> markers = {"o", "square", "diamond"};

	std::vector<Serie> series;
	for (int i = 0; i < paths.size(); i++) {
		Serie serie = read_csv(paths[i]);
		serie.name = names[i];
		serie.color = colors[i];
		serie.marker = markers[i];
		series.push_back(serie);
	}

	create_latex(series, "data/loss_graph.tex");

	return 0;
}
