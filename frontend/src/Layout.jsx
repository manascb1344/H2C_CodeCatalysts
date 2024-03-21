import React from "react";
import Graph from "./Graph";

const Layout = () => {
	return (
		<div class="p-4 sm:ml-64">
			<div class="p-4 border-2 border-gray-200 border-dashed rounded-lg dark:border-gray-700">
				<div class="flex items-center justify-center h-[91vh] mb-4 rounded ">
					<Graph />
				</div>
			</div>
		</div>
	);
};

export default Layout;
