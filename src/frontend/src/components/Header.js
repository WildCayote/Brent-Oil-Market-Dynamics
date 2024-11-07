// frontend/src/components/Header.js

import React from 'react';

const Header = () => {
    return (
        <header className="bg-dark text-white p-3">
            <h1>Brent Oil Price Dashboard</h1>
            <nav>
                <ul className="nav">
                    <li className="nav-item"><a className="nav-link" href="#">Home</a></li>
                    <li className="nav-item"><a className="nav-link" href="#">Analysis</a></li>
                    <li className="nav-item"><a className="nav-link" href="#">About</a></li>
                </ul>
            </nav>
        </header>
    );
}

export default Header;
