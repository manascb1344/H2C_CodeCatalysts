import React, { useState, useEffect, useCallback } from "react";
import ReactFlow, {
	Background,
	Controls,
	applyNodeChanges,
	applyEdgeChanges,
} from "reactflow";
import "reactflow/dist/style.css";

const Graph = () => {
	const [nodes, setNodes] = useState([]);
	const [edges, setEdges] = useState([]);

	const fetchData = useCallback(async () => {
		try {
			const myHeaders = new Headers();
			myHeaders.append("Content-Type", "application/json");
			const raw = JSON.stringify({
				account_number: "8000F5340",
			});
			const requestOptions = {
				method: "POST",
				headers: myHeaders,
				body: raw,
				redirect: "follow",
			};
			const response = await fetch(
				"http://127.0.0.1:5000/transactions",
				requestOptions
			);
			if (!response.ok) {
				throw new Error("Network response was not ok");
			}
			const responseData = await response.json();

			// Extract unique accounts from transactions
			const accounts = new Set();
			const extractAccounts = (transactions) => {
				if (transactions) {
					transactions.forEach((transaction) => {
						accounts.add(transaction.Account);
						accounts.add(transaction["Account.1"]);
					});
				}
			};
			extractAccounts(responseData.transactions_from_account);
			extractAccounts(responseData.transactions_to_account);

			// Prepare nodes
			const initialNodes = Array.from(accounts).map(
				(account, index) => ({
					id: account,
					data: { label: account },
					position: {
						// Position nodes evenly around the center
						x: Math.cos((index / accounts.size) * 2 * Math.PI) * 400,
						y:
							Math.sin((index / accounts.size) * 2 * Math.PI) * 400 -
							300,
					},
					type: "input",
				})
			);

			// Add central node
			initialNodes.push({
				id: "central",
				data: { label: "8000F5340" },
				position: { x: 0, y: 0 },
				type: "input",
			});

			// Prepare edges
			const edgesData = [];
			const extractEdges = (transactions) => {
				transactions.forEach((transaction, i) => {
					if (transaction.Account && transaction["Account.1"]) {
						edgesData.push({
							id: `${transaction.Account}-${transaction["Account.1"]}-${i}`,
							source: transaction.Account,
							target: transaction["Account.1"],
							type: "smoothstep",
							animated: true,
						});
					}
				});
			};
			extractEdges(responseData.transactions_from_account);
			extractEdges(responseData.transactions_to_account);

			const convertData = (data) => {
				return data.map((obj) => ({
					id: obj.id.replace("-", "-"),
					source: obj.source,
					target: obj.target,
					type: "step",
				}));
			};

			// Update state with nodes and edges
			console.log("Nodes:", initialNodes);
			console.log("Edges:", convertData(edgesData));
			setNodes(initialNodes);
			setEdges(convertData(edgesData));
		} catch (error) {
			console.error("Error fetching data:", error);
			setNodes([]);
			setEdges([]);
		}
	}, []);

	useEffect(() => {
		fetchData();
	}, [fetchData]);

	const onNodesChange = useCallback(
		(changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
		[]
	);
	const onEdgesChange = useCallback(
		(changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
		[]
	);

	return (
		<ReactFlow
			nodes={nodes}
			onNodesChange={onNodesChange}
			edges={edges}
			onEdgesChange={onEdgesChange}
			fitView
		>
			<Background />
			<Controls />
		</ReactFlow>
	);
};

export default Graph;

// import { useState, useCallback } from "react";
// import ReactFlow, {
// 	Controls,
// 	Background,
// 	applyNodeChanges,
// 	applyEdgeChanges,
// } from "reactflow";
// import "reactflow/dist/style.css";

// const initialNodes = [
// 	{
// 		id: "1",
// 		data: { label: "Hello" },
// 		position: { x: 0, y: 0 },
// 		type: "input",
// 	},
// 	{
// 		id: "2",
// 		data: { label: "World" },
// 		position: { x: 100, y: 100 },
// 	},
// ];

// const initialEdges = [
// 	{
// 		id: "1-2",
// 		source: "1",
// 		target: "2",
// 		label: "to the",
// 		type: "step",
// 	},
// ];

// function Graph() {
// 	const [nodes, setNodes] = useState(initialNodes);
// 	const [edges, setEdges] = useState(initialEdges);

// 	const onNodesChange = useCallback(
// 		(changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
// 		[]
// 	);
// 	const onEdgesChange = useCallback(
// 		(changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
// 		[]
// 	);

// 	return (
// 		<div style={{ height: "100vh", width: "100vh" }}>
// 			<ReactFlow
// 				nodes={nodes}
// 				onNodesChange={onNodesChange}
// 				edges={edges}
// 				onEdgesChange={onEdgesChange}
// 				fitView
// 			>
// 				<Background />
// 				<Controls />
// 			</ReactFlow>
// 		</div>
// 	);
// }

// export default Graph;
