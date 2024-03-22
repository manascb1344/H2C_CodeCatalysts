import React, { useState, useEffect } from 'react';
import axios from 'axios';
import cheerio from 'cheerio';

const CryptoNewsComponent = () => {
  const [articles, setArticles] = useState([]);

  useEffect(() => {
    const fetchArticles = async () => {
      const cryptoSites = [
        {
            name: "Coindesk",
            address: "https://www.coindesk.com/",
            base: "https://www.coindesk.com",
          },
          {
            name: "CNBCFinance",
            address: "https://www.cnbc.com/finance/",
            base: "https://www.cnbc.com",
          },
          {
            name: "CNBCTech",
            address: "https://www.cnbc.com/technology/",
            base: "https://www.cnbc.com",
          },
          {
            name: "CNNBusiness",
            address: "https://www.cnn.com/business",
            base: "https://www.cnn.com",
          },
          {
            name: "TodayOnChain",
            address: "https://www.todayonchain.com/",
            base: "",
          },
          {
            name: "CryptoSlate",
            address: "https://cryptoslate.com/",
            base: "",
          },
          {
            name: "NewsBTC",
            address: "https://www.newsbtc.com/",
            base: "",
          },
          {
            name: "BitcoinMagazine",
            address: "https://bitcoinmagazine.com/",
            base: "https://bitcoinmagazine.com",
          },
          {
            name: "Bitcoinist",
            address: "https://bitcoinist.com/",
            base: "",
          },
          {
            name: "Forbes",
            address: "https://www.forbes.com/?sh=62d922b92254",
            base: "",
          },
      ];

      const articlesData = [];

      for (const cryptoSite of cryptoSites) {
        try {
          const response = await axios.get(cryptoSite.address);
          const html = response.data;
          const $ = cheerio.load(html);

          $('a:contains("Bitcoin")', html).each(function () {
            const title = $(this).text();
            const url = $(this).attr("href");

            articlesData.push({
              title,
              url: cryptoSite.base + url,
              source: cryptoSite.name,
            });
          });

          $('a:contains("Crypto")', html).each(function () {
            const title = $(this).text();
            const url = $(this).attr("href");

            articlesData.push({
              title,
              url: cryptoSite.base + url,
              source: cryptoSite.name,
            });
          });
        } catch (error) {
          console.error('Error fetching data:', error);
        }
      }

      setArticles(articlesData);
    };

    fetchArticles();
  }, []);

  const itemsPerPage = 4;
  const [currentPage, setCurrentPage] = useState(1);

  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentArticles = articles.slice(startIndex, endIndex);

  const totalPages = Math.ceil(articles.length / itemsPerPage);

  const handlePageChange = (newPage) => {
    setCurrentPage(newPage);
  };

  return (
    <div className="grid h-[100vh] w-full grid-cols-1 md:grid-cols-2 gap-2">
      {currentArticles.map((article, index) => (
        <div
          key={index}
          className="bg-white border border-gray-200 rounded-lg shadow py-8 p-5 transition duration-300 transform hover:scale-103 dark:bg-gray-800 dark:border-gray-700"
        >
          <a href="#">
            <img
              className="rounded-t-lg"
              src="/docs/images/blog/image-1.jpg"
              alt=""
            />
          </a>
          <div className="mt-4">
            <a href="#">
              <h5 className="mb-2 text-2xl font-bold tracking-tight text-gray-900 dark:text-white">
                {article.title}
              </h5>
            </a>
            <p className="mb-3 font-normal text-gray-700 dark:text-gray-400">
              {article.source}
            </p>
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center px-3 py-2 text-sm font-medium text-center text-white bg-blue-700 rounded-lg hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800"
            >
              Read more
              <svg
                className="rtl:rotate-180 w-3.5 h-3.5 ms-2"
                aria-hidden="true"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 14 10"
              >
                <path
                  stroke="currentColor"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M1 5h12m0 0L9 1m4 4L9 9"
                />
              </svg>
            </a>
          </div>
        </div>
      ))}
        <div className="flex h-1/7 justify-center mt-4">
        {Array.from({ length: totalPages }, (_, i) => (
            <button
            key={i}
            className={`px-1 py-1 mx-1 text-sm font-medium text-center text-white bg-blue-700 rounded-lg hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800 ${
                i + 1 === currentPage ? "bg-blue-800" : ""
            }`}
            onClick={() => handlePageChange(i + 1)}
            >
            {i + 1}
            </button>
        ))}
        </div>
    </div>
  

  );
};

export default CryptoNewsComponent;
