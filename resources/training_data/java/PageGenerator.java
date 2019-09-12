package tk.gustavo.pages;

import org.openqa.selenium.WebDriver;

import java.util.concurrent.ConcurrentHashMap;

public class PageGenerator {
    private ConcurrentHashMap<WebDriver, ConcurrentHashMap<String, Object>> pageHashMap =
            new ConcurrentHashMap<WebDriver, ConcurrentHashMap<String, Object>>();

    private synchronized void setPageHashMap(WebDriver driver, String pageName, Object obj) {
        if (!pageHashMap.containsKey(driver)) {
            ConcurrentHashMap<String, Object> pageHashMapPageNamePageObj = new ConcurrentHashMap<String, Object>();
            pageHashMapPageNamePageObj.put(pageName, obj);
            pageHashMap.put(driver, pageHashMapPageNamePageObj);
        } else {
            if (!pageHashMap.get(driver).containsKey(pageName)) {
                pageHashMap.get(driver).put(pageName, obj);
            }
        }
    }

    private synchronized Object getPageHashMap(WebDriver driver, String pageName) {
        if (pageHashMap.containsKey(driver)) {
            return pageHashMap.get(driver).get(pageName);
        } else {
            return null;
        }
    }


}
