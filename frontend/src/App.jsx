import React from "react";
import "./App.css";
import SideNav from "./components/SideNav";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Graph from "./Graph.jsx";
import Dash from "./Dash.jsx";
import TP from "./TP.jsx";
import TransactionAnalysis from "./TransactionAnalysis.jsx";

function App() {
	return (
		<BrowserRouter>
			<div className="side-nav">
				<SideNav />
			</div>
			<div className="content">
				<div className="p-4 sm:ml-64">
					<div className="p-4 border-2 border-gray-200 border-dashed rounded-lg dark:border-gray-700">
						<div className="flex items-center justify-center h-[91vh] mb-4 rounded ">
							<Routes>
								<Route path="/graph" element={<Graph />} />
								<Route path="/dashboard" element={<Dash />} />
								<Route path="/tp" element={<TP />} />
								<Route path="/transaction-analysis" element={<TransactionAnalysis />} />
							</Routes>
						</div>
					</div>
				</div>
			</div>
		</BrowserRouter>
	);
}

export default App;
