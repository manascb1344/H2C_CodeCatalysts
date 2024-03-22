import React from "react";
import "./App.css";
import SideNav from "./components/SideNav";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Graph from "./Graph.jsx";
import Dash from "./Dash.jsx";
import TP from "./TP.jsx";
import TradingViewWidget from "./Chart.jsx"

function App() {
	return (
		<BrowserRouter>
			<div className="side-nav">
				<SideNav />
			</div>
			<div className="content">
				<div class="p-4 sm:ml-64">
					<div class="p-4 border-2 border-gray-200 border-dashed rounded-lg dark:border-gray-700">
						<div class="flex items-center justify-center h-[91vh] mb-4 rounded ">
							<Routes>
								<Route path="/graph" element={<Graph />} />
								<Route path="/dashboard" element={<Dash />} />
								<Route path="/tp" element={<TP />} />
								<Route path="/charts" element={<TradingViewWidget />} />
							</Routes>
						</div>
					</div>
				</div>
			</div>
		</BrowserRouter>
	);
}

export default App;
