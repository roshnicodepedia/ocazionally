# Ocazionally Data Schema (v1)

One schema is used by the vendor sheet, the synthetic generator, MongoDB and any future app.
Every table has a `source` field: `real` or `synthetic`.

## vendors
| field | type | notes |
|---|---|---|
| vendor_id | string | e.g. V0001 |
| name | string | |
| category | string | decor, catering, sound_dj, photography, mehendi, venue, lighting, other |
| city | string | |
| area | string | locality within the city |
| price_min | int | INR, lowest typical quote |
| price_max | int | INR, highest typical quote |
| capacity_per_month | int | events they can handle in a month |
| peak_months | string | e.g. "10,11,12,1,2" |
| years_active | int | |
| rating_avg | float | 0 if no ratings yet (new vendor) |
| onboarded_date | date | |
| verification_status | string | pending, verified, rejected (set by admin) |
| source | string | real / synthetic |

## customers (anonymised, no names or phone numbers)
| field | type | notes |
|---|---|---|
| customer_id | string | hashed id |
| customer_type | string | individual, housing_society, college, corporate |
| city | string | |

## events_log (every platform action, one row per action)
| field | type | notes |
|---|---|---|
| log_id | string | |
| timestamp | timestamp | |
| action | string | search, vendor_view, quote_request, quote_response, availability_update, booking_confirmed, booking_cancelled, review_submitted, vendor_verified, booking_assigned |
| actor_role | string | customer, vendor or admin: who performed the action |
| session_id | string | |
| customer_id | string | |
| vendor_id | string | null for searches |
| city | string | |
| event_type | string | wedding, navratri_society, diwali, birthday, college_fest, corporate, uttarayan, other |
| budget | int | INR, only for searches / quote requests |
| guest_count | int | |
| event_date | date | date of the planned event |

## bookings
| field | type | notes |
|---|---|---|
| booking_id | string | |
| event_instance_id | string | groups all bookings of the same real event (used for FP-Growth bundles) |
| customer_id | string | |
| vendor_id | string | |
| event_type | string | as above |
| city | string | |
| guest_count | int | |
| booking_date | date | when it was booked |
| event_date | date | when the event happens |
| amount | int | INR |
| status | string | confirmed, completed, cancelled |
| cancelled_by | string | null, customer, vendor |
| response_time_minutes | int | how long the vendor took to respond to the quote request |
| source | string | real / synthetic |

## reviews
| field | type | notes |
|---|---|---|
| review_id | string | |
| booking_id | string | |
| vendor_id | string | |
| rating | int | 1 to 5 |
| review_text | string | optional |
| timestamp | timestamp | |
| source | string | real / synthetic |

## Privacy rules (Unit 6 ethics section)
- No names, phone numbers or addresses of customers in analytics tables
- customer_id is a hash
- Keep only fields needed for analysis
