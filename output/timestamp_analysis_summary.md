# Timestamp Type Analysis: Twitter vs News

## Summary

Analysis of timestamp columns in Twitter and News datasets to understand their formats and characteristics.

---

## Twitter Dataset (`all_twitter_posts_merged_with_details.csv`)

### Timestamp Columns Found:

#### 1. `created_at` (Tweet Posting Time)
- **Purpose**: When the tweet was posted
- **Format**: `2024-07-24T22:07:58.000Z`
- **Data Type**: String (object)
- **Completeness**: 100% (3,843/3,843 rows)
- **Date Range**: 2024-07-24 to 2024-12-31 (160 days)
- **Characteristics**:
  - ISO 8601 format with UTC timezone (Z suffix)
  - Includes milliseconds (`.000`)
  - Variable time of day (not always midnight)
  - String length: 24 characters
  - 3,837 unique timestamps

#### 2. `author_created_at` (Author Account Creation Time)
- **Purpose**: When the Twitter author account was created
- **Format**: `2009-10-18T03:37:03.000Z`
- **Data Type**: String (object)
- **Completeness**: 93.44% (3,591/3,843 rows)
- **Date Range**: 2006-11-01 to 2024-12-27 (6,631 days / ~18 years)
- **Characteristics**:
  - ISO 8601 format with UTC timezone (Z suffix)
  - Includes milliseconds (`.000`)
  - Variable time of day
  - String length: 24 characters
  - 2,908 unique account creation dates
  - **Note**: Not suitable for temporal analysis of tweet content, but useful for account age analysis

---

## News Dataset (`all_newspaper_articles.csv`)

### Timestamp Columns Found:

#### 1. `article_date` (Article Publication Date)
- **Purpose**: When the article was published
- **Format**: `2022-02-12T00:00:00Z`
- **Data Type**: String (object)
- **Completeness**: 100% (2,577/2,577 rows)
- **Date Range**: 2015-01-07 to 2024-12-18 (3,633 days / ~10 years)
- **Characteristics**:
  - ISO 8601 format with UTC timezone (Z suffix)
  - **No milliseconds** (unlike Twitter)
  - **Always midnight** (`00:00:00`) - date-level precision only
  - String length: 20 characters (shorter than Twitter)
  - 981 unique publication dates

---

## Key Differences

| Aspect | Twitter `created_at` | News `article_date` |
|--------|----------------------|---------------------|
| **Format** | `YYYY-MM-DDTHH:MM:SS.000Z` | `YYYY-MM-DDTHH:MM:SSZ` |
| **Milliseconds** | Yes (`.000`) | No |
| **Time Precision** | Hour/minute/second | Date only (always `00:00:00`) |
| **String Length** | 24 characters | 20 characters |
| **Temporal Coverage** | 2024-07-24 to 2024-12-31 (5.5 months) | 2015-01-07 to 2024-12-18 (10 years) |
| **Use Case** | Real-time tweet posting analysis | Daily article publication analysis |

---

## Parsing Compatibility

Both formats are **fully compatible** with pandas `pd.to_datetime()`:
- Both use ISO 8601 standard
- Both include UTC timezone indicator (`Z`)
- Both parse correctly with `pd.to_datetime(..., errors='coerce', utc=True)`

**Example parsing:**
```python
# Twitter
twitter_dates = pd.to_datetime(df_twitter['created_at'], errors='coerce', utc=True)
# Result: 100% success rate

# News  
news_dates = pd.to_datetime(df_news['article_date'], errors='coerce', utc=True)
# Result: 100% success rate
```

---

## Recommendations for Temporal Analysis

1. **For tweet-level temporal analysis**: Use `created_at` (100% complete, precise timestamps)
2. **For article-level temporal analysis**: Use `article_date` (100% complete, date-level precision)
3. **For account age analysis**: Use `author_created_at` (93.44% complete, but not for content timing)
4. **For cross-source comparison**: Both can be aggregated to daily/hourly periods, but note that News only has date-level precision

---

## Notes

- Twitter data spans a shorter time period (5.5 months in 2024) but has precise timestamps
- News data spans a much longer period (10 years: 2015-2024) but only has date-level precision
- Both sources have 100% complete primary timestamp columns
- The format differences are minor and don't affect parsing compatibility
