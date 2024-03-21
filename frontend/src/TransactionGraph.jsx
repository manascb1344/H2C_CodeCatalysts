import React, { useEffect, useRef } from "react";
import { DataSet, Network } from "vis-network";

const TransactionGraph = ({ transactions }) => {
	const containerRef = useRef(null);
	const networkRef = useRef(null);

	useEffect(() => {
		if (!containerRef.current || !transactions) return;

		const nodes = new DataSet([
			{ id: "center", label: "Center Node", color: "blue" }, // Center node
			...Array.from(
				new Set(
					transactions.flatMap((transaction) => [
						transaction.Account,
						transaction["Account.1"],
					])
				)
			).map((account) => ({ id: account, label: account })),
		]);

		const edges = new DataSet(
			transactions.map((transaction) => ({
				from: transaction.Account,
				to: transaction["Account.1"],
			}))
		);

		const data = { nodes, edges };

		const options = {
			nodes: {
				shape: "dot",
				size: 20,
				font: {
					size: 15,
				},
			},
			edges: {
				arrows: {
					to: {
						enabled: true,
						scaleFactor: 0.5,
					},
				},
			},
			physics: {
				enabled: true,
			},
		};

		const network = new Network(containerRef.current, data, options);
		networkRef.current = network;

		return () => {
			if (networkRef.current !== null) {
				networkRef.current.destroy();
				networkRef.current = null;
			}
		};
	}, [transactions]);

	return (
		<div
			ref={containerRef}
			style={{ width: "100%", height: "600px" }}
		/>
	);
};

export default TransactionGraph;
