// frontend/src/components/EventFilter.js

import React, { useEffect, useState } from 'react';
import DatePicker from 'react-datepicker';
import "react-datepicker/dist/react-datepicker.css";
import { Button, Form } from 'react-bootstrap';
import axios from 'axios';

const EventFilter = ({ onFilter }) => {
    const [startDate, setStartDate] = useState(null);
    const [endDate, setEndDate] = useState(null);
    const [events, setEvents] = useState([]);

   

    const handleFilter = () => {
        onFilter(startDate, endDate);
    };

    return (
        <div className="mb-3">
            <h3>Filter Events</h3>
            <DatePicker selected={startDate} onChange={date => setStartDate(date)} placeholderText="Start Date" />
            <DatePicker selected={endDate} onChange={date => setEndDate(date)} placeholderText="End Date" />
            
            <Button variant="primary" onClick={handleFilter}>Apply Filter</Button>
        </div>
    );
}

export default EventFilter;
